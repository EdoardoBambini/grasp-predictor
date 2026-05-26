"""Rank DROID test episodes by their P(failure) trace and pick demo candidates for the grasp-failure video."""
from __future__ import annotations

import glob
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from run_training_cached import split_by_pools_proportional  # noqa: E402
from video_overlay_demo import (  # noqa: E402
    SEQ_LEN, load_model, _prep_kin, _prep_cnn, prefix_bag_proba,
)

TAIL_FRACTION = 0.30  # matches droid_tail_fraction the model was trained on


def bag_proba(model, pk_list, pc_list, lo, hi):
    """Bag-level P(failure) over windows [lo:hi] (a contiguous sub-bag)."""
    n = hi - lo
    xk = np.stack(pk_list[lo:hi])[None]   # (1, n, T, kin_dim)
    xc = np.stack(pc_list[lo:hi])[None]   # (1, n, T, cnn_dim)
    mask = torch.ones((1, n), dtype=torch.bool)
    nl = torch.zeros((1, 512), dtype=torch.float32)
    is_d = torch.ones((1,), dtype=torch.bool)
    probs, _ = model.predict_proba(torch.from_numpy(xk), torch.from_numpy(xc),
                                   mask, nl_emb=nl, is_droid=is_d)
    return float(probs[0].item())


def tail_sliding_trace(model, pk_list, pc_list):
    """Trailing-window bag trace: at window-count i, pool the last
    K=max(1,round(0.3*M)) windows ending at i. Ends exactly at the model's
    tail-30% training bag, so it reflects the in-distribution prediction."""
    M = len(pk_list)
    K = max(1, round(TAIL_FRACTION * M))
    return [bag_proba(model, pk_list, pc_list, max(0, i - K), i)
            for i in range(2, M + 1)], K


CACHE_GLOB = str(ROOT / "results" / "cnn_cache_spatial" / "droid" / "*.npz")
POOLS_JSON = str(ROOT / "results" / "pools_via_mosaico.json")
MODEL_DIR = str(ROOT / "results" / "multimodal_indist_v10l_seed42")
PARQUET_ROOT = Path("E:/datasets/droid/data")
VIDEO_ROOT = Path("E:/datasets/droid/videos/observation.images.exterior_1_left")
OUT_JSON = ROOT / "results" / "video_overlay_demo" / "candidate_traces_tail.json"

T_MIN, T_MAX = 300, 900   # 6-18 s at 50 Hz
MIN_WINDOWS = 4           # need enough windows for a meaningful trace

# FAILURE dynamic-trace thresholds
F_PSTART_MAX = 0.30
F_PEND_MIN = 0.60
F_RISE_MIN = 0.40
# relaxed fallback if fewer than 2 strict failures
F_PEND_MIN_RELAX = 0.55
F_RISE_MIN_RELAX = 0.35
# SUCCESS stable-low threshold
S_MAXP_MAX = 0.20


def parquet_episode_sets():
    """episode_index set per shard stem, read once (defensive renderability)."""
    import pyarrow.parquet as pq
    sets = {}
    for pqf in sorted(PARQUET_ROOT.glob("chunk-000/*.parquet")):
        col = pq.read_table(str(pqf), columns=["episode_index"]).column(0)
        sets[pqf.stem] = set(col.to_pylist())
    return sets


def trace_metrics(trace):
    t = np.asarray(trace, dtype=np.float64)
    k = max(1, len(t) // 3)
    p_start = float(t[:k].mean())
    p_end = float(t[-2:].mean()) if len(t) >= 2 else float(t[-1])
    final = float(t[-1])
    rise = p_end - p_start
    max_p, min_p = float(t.max()), float(t.min())
    idx_low = next((i for i, p in enumerate(t) if p > 0.3), None)
    idx_high = next((i for i, p in enumerate(t) if p >= 0.6), None)
    if idx_low is not None and idx_high is not None and idx_high >= idx_low:
        spread = idx_high - idx_low + 1
    elif idx_high is not None:
        spread = 1
    else:
        spread = 0
    band_count = int(np.sum((t >= 0.3) & (t <= 0.6)))
    return dict(p_start=p_start, p_end=p_end, final=final, rise=rise,
                max_p=max_p, min_p=min_p, spread=spread, band_count=band_count)


def main():
    print("[1/4] reconstructing v10l test split (seed 42, droid)")
    cache = {"droid": sorted(glob.glob(CACHE_GLOB))}
    print(f"  cache: {len(cache['droid'])} droid episodes")
    test = split_by_pools_proportional(
        cache, POOLS_JSON, success_ratio_train=0.65, seed=42)[3]["droid"]
    print(f"  test_per_ds['droid']: {len(test)} episodes")

    with open(POOLS_JSON) as f:
        pools = json.load(f)
    succ_set = set(pools["droid"].get("success", []))
    fail_set = set(pools["droid"].get("failure", []))

    print("[2/4] reading parquet episode_index sets")
    ep_sets = parquet_episode_sets()
    print(f"  shards with parquet: {sorted(ep_sets)}")

    print("[3/4] loading model")
    bundle = load_model(MODEL_DIR)
    model, kin_mean, kin_std, cnn_mean, cnn_std, kin_dim, cnn_dim = bundle

    print("[4/4] scanning candidates")
    failures, successes, skipped = [], [], 0
    for n, npz_path in enumerate(test):
        stem = Path(npz_path).stem  # droid_file-XXX_epYYYY
        name = stem
        if name in fail_set:
            label = "FAILURE"
        elif name in succ_set:
            label = "SUCCESS"
        else:
            skipped += 1
            continue

        ep_id = int(stem.split("_ep")[-1])
        file_stem = stem.split("_ep")[0].replace("droid_", "")
        # renderability guards (all true here, kept defensively)
        if file_stem not in ep_sets or ep_id not in ep_sets[file_stem]:
            skipped += 1
            continue
        if not (VIDEO_ROOT / "chunk-000" / f"{file_stem}.mp4").exists():
            skipped += 1
            continue

        with np.load(npz_path) as cache_npz:
            kin_seq = cache_npz["kin"].astype(np.float32)
            T = len(kin_seq)
            if not (T_MIN <= T <= T_MAX):
                continue
            cnn_seq = cache_npz["cnn"]
            win_starts = list(range(0, T - SEQ_LEN + 1, SEQ_LEN))
            M = len(win_starts)
            if M < MIN_WINDOWS:
                continue
            pk_list = [_prep_kin(kin_seq[s:s + SEQ_LEN], kin_mean, kin_std, kin_dim)
                       for s in win_starts]
            pc_list = [_prep_cnn(cnn_seq[s:s + SEQ_LEN], cnn_mean, cnn_std)
                       for s in win_starts]

        with torch.no_grad():
            trace, K = tail_sliding_trace(model, pk_list, pc_list)
            prefix_final = (prefix_bag_proba(model, pk_list, pc_list, M)
                            if label == "FAILURE" else None)

        m = trace_metrics(trace)
        rec = dict(name=name, label=label, T=int(T), M=int(M), K=int(K),
                   prefix_final=(round(prefix_final, 4) if prefix_final is not None else None),
                   trace=[round(float(x), 4) for x in trace], **m)
        (failures if label == "FAILURE" else successes).append(rec)
        if (n + 1) % 25 == 0:
            print(f"  ...{n + 1}/{len(test)} scanned "
                  f"(F={len(failures)} S={len(successes)})")

    print(f"\nscanned: F={len(failures)} S={len(successes)} "
          f"(skipped/no-label/no-source/out-of-T={skipped} + T/M filtered)")

    # ---- rank FAILURES ----
    for r in failures:
        r["passes_strict"] = (r["p_start"] < F_PSTART_MAX and
                              r["p_end"] > F_PEND_MIN and r["rise"] > F_RISE_MIN)
        r["passes_relax"] = (r["p_start"] < F_PSTART_MAX and
                             r["p_end"] > F_PEND_MIN_RELAX and
                             r["rise"] > F_RISE_MIN_RELAX)
    strict_f = [r for r in failures if r["passes_strict"]]
    pool_f = strict_f if len(strict_f) >= 2 else [r for r in failures if r["passes_relax"]]
    pool_f.sort(key=lambda r: (-(r["spread"] >= 2), -r["rise"], -r["spread"]))
    used_relax = len(strict_f) < 2

    # ---- rank SUCCESSES ----
    pool_s = [r for r in successes if r["max_p"] < S_MAXP_MAX]
    pool_s.sort(key=lambda r: r["max_p"])

    fail_ends = [r["p_end"] for r in failures]
    if fail_ends:
        print(f"\nsanity: mean tail-bag p_end over failures = "
              f"{sum(fail_ends)/len(fail_ends):.3f} (official tail eval ~0.87)")

    def show(r):
        pf = f" prefix_final={r['prefix_final']:.3f}" if r.get("prefix_final") is not None else ""
        return (f"  {r['name']:<26} T={r['T']:>3} M={r['M']:>2} K={r.get('K','?')} "
                f"p_start={r['p_start']:.3f} p_end={r['p_end']:.3f} "
                f"rise={r['rise']:+.3f} spread={r['spread']} "
                f"band={r['band_count']} max={r['max_p']:.3f}{pf}\n"
                f"      trace={['%.3f' % x for x in r['trace']]}")

    print("\n================ TOP FAILURE CANDIDATES "
          f"({'RELAXED' if used_relax else 'strict'}) ================")
    for r in pool_f[:12]:
        print(show(r))
    print("\n================ TOP SUCCESS CANDIDATES (stable-low) ================")
    for r in pool_s[:12]:
        print(show(r))

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_JSON, "w") as f:
        json.dump(dict(
            test_size=len(test),
            n_failures_scanned=len(failures),
            n_successes_scanned=len(successes),
            used_relaxed_failures=used_relax,
            failure_candidates=pool_f,
            success_candidates=pool_s,
            all_failures=sorted(failures, key=lambda r: -r["rise"]),
            all_successes=sorted(successes, key=lambda r: r["max_p"]),
        ), f, indent=2)
    print(f"\nwrote {OUT_JSON}")


if __name__ == "__main__":
    sys.exit(main())
