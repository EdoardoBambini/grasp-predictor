"""Smoke test for the released LateFusionLSTMMIL checkpoint: load, param-count, forward, predict_proba."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from models.mil_attention import LateFusionLSTMMIL

MODEL_PATH = REPO / "results" / "multimodal_indist_v10l_seed7" / "checkpoints" / "best_model.pt"

KIN_DIM = 16
CNN_DIM = 360
SEQ_LEN = 50


def build_released_model() -> LateFusionLSTMMIL:
    # Released configuration, mirrors scripts/train.sh (--hidden 256, uni LSTM,
    # gated MIL pool, pre-pool NL concat 64-D). hidden//4 = 64 LSTM units,
    # hidden//2 = 128 pool/head width.
    return LateFusionLSTMMIL(
        kin_dim=KIN_DIM, cnn_dim=CNN_DIM,
        kin_hidden=64, vis_hidden=64,
        num_layers=1, dropout=0.35,
        bidirectional=False,
        attn_pool=True,
        mil_attn_hidden=128, head_hidden=128,
        pool_mode="gated",
        use_nl_concat=True, nl_concat_dim=64,
    )


@pytest.fixture(scope="module")
def model():
    if not MODEL_PATH.exists():
        pytest.skip(f"Released checkpoint missing: {MODEL_PATH}")
    m = build_released_model()
    state = torch.load(MODEL_PATH, map_location="cpu", weights_only=True)
    # Tolerate an SWA-wrapped checkpoint (AveragedModel adds a "module." prefix
    # and an "n_averaged" buffer).
    state = {(k[len("module."):] if k.startswith("module.") else k): v
             for k, v in state.items()}
    state.pop("n_averaged", None)
    m.load_state_dict(state)
    m.eval()
    return m


def test_param_count(model):
    n = sum(p.numel() for p in model.parameters() if p.requires_grad)
    assert n == 237635, f"Expected 237635 trainable params, got {n}"


def test_forward_shape(model):
    bags, windows = 2, 3
    x_kin = torch.randn(bags, windows, SEQ_LEN, KIN_DIM)
    x_cnn = torch.randn(bags, windows, SEQ_LEN, CNN_DIM)
    mask = torch.ones(bags, windows, dtype=torch.bool)
    with torch.no_grad():
        logits, attn = model(x_kin, x_cnn, mask=mask)
    assert logits.shape == (bags,), f"Expected ({bags},) logits, got {tuple(logits.shape)}"
    assert attn.shape == (bags, windows), f"Expected ({bags},{windows}) attn, got {tuple(attn.shape)}"


def test_predict_proba_range(model):
    bags, windows = 2, 3
    x_kin = torch.randn(bags, windows, SEQ_LEN, KIN_DIM)
    x_cnn = torch.randn(bags, windows, SEQ_LEN, CNN_DIM)
    mask = torch.ones(bags, windows, dtype=torch.bool)
    p, _ = model.predict_proba(x_kin, x_cnn, mask=mask)
    p = p.numpy()
    assert ((p >= 0.0) & (p <= 1.0)).all(), f"P out of [0, 1]: {p}"
