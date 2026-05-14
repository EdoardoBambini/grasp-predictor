"""Smoke test for the released LateFusionLSTM checkpoint.

Loads the trained weights and verifies:
- The model loads without errors
- The trainable parameter count matches the released number (108 547)
- A forward pass on random tensors produces the expected logit shape
- predict_proba returns values in [0, 1]

Intentionally lightweight: needs only torch + numpy + pytest (not the full
SDK). Designed to run in CI on a clean Linux image with the shipped
best_model.pt as the only data dependency.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from models.cached_lstm import LateFusionLSTM

MODEL_PATH = REPO / "results" / "multimodal_indist_v9_sharp" / "checkpoints" / "best_model.pt"


@pytest.fixture(scope="module")
def model():
    if not MODEL_PATH.exists():
        pytest.skip(f"Released checkpoint missing: {MODEL_PATH}")
    m = LateFusionLSTM(
        kin_dim=15, cnn_dim=64,
        kin_hidden=64, vis_hidden=64,
        num_layers=1, dropout=0.35, bidirectional=True,
        fc_dropout=0.35, attn_pool=True,
    )
    state = torch.load(MODEL_PATH, map_location="cpu", weights_only=True)
    m.load_state_dict(state)
    m.eval()
    return m


def test_param_count(model):
    n = sum(p.numel() for p in model.parameters() if p.requires_grad)
    assert n == 108547, f"Expected 108547 trainable params, got {n}"


def test_forward_shape(model):
    batch = 4
    x_kin = torch.randn(batch, 50, 15)
    x_cnn = torch.randn(batch, 50, 64)
    with torch.no_grad():
        logit = model(x_kin, x_cnn)
    assert logit.shape == (batch,), f"Expected ({batch},) logit, got {tuple(logit.shape)}"


def test_predict_proba_range(model):
    x_kin = torch.randn(2, 50, 15)
    x_cnn = torch.randn(2, 50, 64)
    p = model.predict_proba(x_kin, x_cnn).numpy()
    assert ((p >= 0.0) & (p <= 1.0)).all(), f"P out of [0, 1]: {p}"
