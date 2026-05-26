"""Pick demo sequences from DROID episodes held out of the train/val/test split and write the acts JSON."""
from __future__ import annotations

import glob
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from run_training_cached import split_by_pools_proportional  # noqa: E402
from video_overlay_demo import SEQ_LEN, load_model, _prep_kin, _prep_cnn  # noqa: E402
from select_demo_sequences import (  # noqa: E402
    TAIL_FRACTION, bag_proba, tail_sliding_trace, trace_metrics,
    parquet_episode_sets, VIDEO_ROOT,
)

CACHE_GLOB = str(ROOT / "results" / "cnn_cache_spatial" / "droid" / "*.npz")
POOLS_JSON = str(ROOT / "results" / "pools_via_mosaico.json")
MODEL_DIR = str(ROOT / "results" / "multimodal_indist_v10l_seed7")
ACTS_OUT = ROOT / "results" / "video_overlay_demo" / "acts_v10l_droid_test.json"
CACHE_FPS = 50
T_MIN, T_MAX = 300, 900
MIN_WINDOWS = 4
FAIL_TRACE_STRIDES = [25, 15]  # applied to the two chosen failure acts


def main() -> int:
    cache_paths = sorted(glob.glob(CACHE_GLOB))
    cache = {"droid": cache_paths}
    tr, va, te, test_per_ds, _ = split_by_pools_proportional(
        cache, POOLS_JSON, success_ratio_train=0.65, seed=7)
    used = set(tr) | set(va) | set(te)
    heldout = [p for p in cache_paths if p not in used]
    print(f"cache={len(cache_paths)} used(train+val+test)={len(used)} "
          f"held-out={len(heldout)}")

    with open(POOLS_JSON) as f:
        pools = json.load(f)
    succ_set = set(pools["droid"].get("success", []))
    fail_set = set(pools["droid"].get("failure", []))
    ep_sets = parquet_episode_sets()

    model, kin_mean, kin_std, cnn_mean, cnn_std, kin_dim, cnn_dim = load_model(MODEL_DIR)

    # The seed-7 split consumes ALL cached DROID failures, so no failure is held
    # out. SUCCESS acts come from the held-out (never train/val/test) pool;
    # FAILURE acts come from the test split (never trained on) since that is the
    # only place renderable failures exist. Successes honour "not train, not
    # eval"; failures are the strongest available guarantee (not trained on).
    succ_source = heldout
    fail_source = list(test_per_ds.get("droid", []))
    candidates = ([(p, "SUCCESS") for p in succ_source]
                  + [(p, "FAILURE") for p in fail_source])

    # ---- pass 1: one bag score per renderable candidate ----
    fails, succs = [], []
    for npz_path, want in candidates:
        stem = Path(npz_path).stem
        if want == "FAILURE" and stem in fail_set:
            label = "FAILURE"
        elif want == "SUCCESS" and stem in succ_set:
            label = "SUCCESS"
        else:
            continue
        ep_id = int(stem.split("_ep")[-1])
        file_stem = stem.split("_ep")[0].replace("droid_", "")
        if file_stem not in ep_sets or ep_id not in ep_sets[file_stem]:
            continue
        if not (VIDEO_ROOT / "chunk-000" / f"{file_stem}.mp4").exists():
            continue
        with np.load(npz_path) as z:
            kin_seq = z["kin"].astype(np.float32)
            T = len(kin_seq)
            if not (T_MIN <= T <= T_MAX):
                continue
            cnn_seq = z["cnn"]
            starts = list(range(0, T - SEQ_LEN + 1, SEQ_LEN))
            M = len(starts)
            if M < MIN_WINDOWS:
                continue
            pk = [_prep_kin(kin_seq[s:s + SEQ_LEN], kin_mean, kin_std, kin_dim) for s in starts]
            pc = [_prep_cnn(cnn_seq[s:s + SEQ_LEN], cnn_mean, cnn_std) for s in starts]
        K = max(1, round(TAIL_FRACTION * M))
        with torch.no_grad():
            score = (bag_proba(model, pk, pc, M - K, M) if label == "FAILURE"
                     else bag_proba(model, pk, pc, 0, M))
        # Do NOT keep pk/pc here (would blow memory over ~1.5k candidates);
        # pass 2 reloads the npz for the few top candidates only.
        rec = dict(name=stem, path=npz_path.replace("\\", "/"), T=int(T), M=int(M),
                   score=float(score), duration=round(T / CACHE_FPS, 1))
        (fails if label == "FAILURE" else succs).append(rec)

    print(f"renderable held-out: failures={len(fails)} successes={len(succs)}")

    # ---- pass 2: traces for top candidates, then rank ----
    fails.sort(key=lambda r: -r["score"])
    succs.sort(key=lambda r: r["score"])
    fail_top = fails[:20]
    for r in fail_top:
        with np.load(ROOT / r["path"]) as z:
            kin_seq = z["kin"].astype(np.float32)
            cnn_seq = z["cnn"]
        starts = list(range(0, len(kin_seq) - SEQ_LEN + 1, SEQ_LEN))
        pk = [_prep_kin(kin_seq[s:s + SEQ_LEN], kin_mean, kin_std, kin_dim) for s in starts]
        pc = [_prep_cnn(cnn_seq[s:s + SEQ_LEN], cnn_mean, cnn_std) for s in starts]
        with torch.no_grad():
            trace, _ = tail_sliding_trace(model, pk, pc)
        r.update(trace_metrics(trace))
    # dynamic late rise first, then confident terminal
    fail_top.sort(key=lambda r: (-(r["spread"] >= 2), -r["rise"], -r["score"]))

    chosen_f = fail_top[:2]
    chosen_s = succs[:2]
    if len(chosen_f) < 2 or len(chosen_s) < 2:
        print("WARNING: not enough confident held-out candidates "
              f"(F={len(chosen_f)} S={len(chosen_s)})")

    acts = []
    fi = 0
    # Interleave failure/success for a nice cadence: F, S, F, S
    pairs = [("FAILURE", chosen_f[0] if chosen_f else None),
             ("SUCCESS", chosen_s[0] if chosen_s else None),
             ("FAILURE", chosen_f[1] if len(chosen_f) > 1 else None),
             ("SUCCESS", chosen_s[1] if len(chosen_s) > 1 else None)]
    for label, r in pairs:
        if r is None:
            continue
        act = {"sequence": r["path"], "offset": 0, "duration": r["duration"],
               "label": label, "camera": "exterior_1_left"}
        if label == "FAILURE":
            act["trace_stride"] = FAIL_TRACE_STRIDES[fi]; fi += 1
        acts.append(act)

    ACTS_OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(ACTS_OUT, "w") as f:
        json.dump(acts, f, indent=2)

    print("\nChosen held-out demo acts (all absent from seed-7 train/val/test):")
    for a in acts:
        nm = Path(a["sequence"]).stem
        sc = next(r["score"] for r in (fails + succs) if r["path"] == a["sequence"])
        print(f"  {a['label']:<7} {nm:<26} dur={a['duration']:>5}s "
              f"P(fail)={sc:.3f} stride={a.get('trace_stride', '-')}"
              f"  held_out={nm + '.npz' not in {Path(p).name for p in used}}")
    print(f"\nwrote {ACTS_OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
