"""
LSTMs that consume pre-computed CNN embeddings + kinematic features. The
visual backbone is frozen and runs once during the CNN pre-cache, so this
module contains only the temporal head.

Input: x_kin (B, T, 15) float32, x_cnn (B, T, cnn_dim) float32.
Output: logits (B,). Pair with BCEWithLogitsLoss or FocalLoss.
"""
from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor


class CachedFeatureLSTM(nn.Module):
    def __init__(
        self,
        kin_dim: int = 15,
        cnn_dim: int = 576,
        hidden_size: int = 256,
        num_layers: int = 2,
        dropout: float = 0.3,
        bidirectional: bool = True,
        fc_dropout: float = 0.2,
    ) -> None:
        super().__init__()
        directions = 2 if bidirectional else 1
        self._lstm = nn.LSTM(
            input_size=kin_dim + cnn_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=bidirectional,
            batch_first=True,
        )
        self._dropout = nn.Dropout(p=fc_dropout)
        self._head = nn.Linear(hidden_size * directions, 1)

    def forward(self, x_kin: Tensor, x_cnn: Tensor) -> Tensor:
        fused = torch.cat([x_kin, x_cnn], dim=-1)  # (B, T, 591)
        lstm_out, _ = self._lstm(fused)
        last = lstm_out[:, -1, :]
        return self._head(self._dropout(last)).squeeze(-1)

    @torch.no_grad()
    def predict_proba(self, x_kin: Tensor, x_cnn: Tensor) -> Tensor:
        self.eval()
        return torch.sigmoid(self.forward(x_kin, x_cnn))


class LateFusionLSTM(nn.Module):
    """Late fusion BiLSTM: kin and visual streams pass through separate LSTMs.
    Their pooled outputs are concatenated and fed to a linear head.

    Streams are fused at the head, not at input, so each modality builds an
    independent temporal representation. Pair with an IPCA-reduced visual
    stream so the two stream widths are dimensionally balanced.
    """

    def __init__(
        self,
        kin_dim: int = 15,
        cnn_dim: int = 20,  # post-IPCA recommended
        kin_hidden: int = 64,
        vis_hidden: int = 64,
        num_layers: int = 1,
        dropout: float = 0.3,
        bidirectional: bool = True,
        fc_dropout: float = 0.3,
        attn_pool: bool = False,
    ) -> None:
        super().__init__()
        directions = 2 if bidirectional else 1
        self._lstm_kin = nn.LSTM(
            input_size=kin_dim, hidden_size=kin_hidden,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=bidirectional, batch_first=True,
        )
        self._lstm_vis = nn.LSTM(
            input_size=cnn_dim, hidden_size=vis_hidden,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=bidirectional, batch_first=True,
        )
        self._dropout = nn.Dropout(p=fc_dropout)
        fused_dim = (kin_hidden + vis_hidden) * directions
        self._head = nn.Linear(fused_dim, 1)

        # Optional learned softmax attention over the time dimension, replacing
        # the last-timestep readout. Episode-level labels (DROID, Fractal) make
        # any frame in the window informative, so a learned pool generalizes
        # better than fixed last-timestep selection.
        self._attn_pool = attn_pool
        if attn_pool:
            self._attn_kin = nn.Linear(kin_hidden * directions, 1)
            self._attn_vis = nn.Linear(vis_hidden * directions, 1)
        else:
            self._attn_kin = None
            self._attn_vis = None

    def forward(self, x_kin: Tensor, x_cnn: Tensor) -> Tensor:
        # Independent temporal encoding per modality
        kin_out, _ = self._lstm_kin(x_kin)   # (B, T, kin_hidden*dir)
        vis_out, _ = self._lstm_vis(x_cnn)   # (B, T, vis_hidden*dir)
        if self._attn_pool:
            a_kin = torch.softmax(self._attn_kin(kin_out), dim=1)  # (B, T, 1)
            kin_pooled = (a_kin * kin_out).sum(dim=1)              # (B, kin_hidden*dir)
            a_vis = torch.softmax(self._attn_vis(vis_out), dim=1)
            vis_pooled = (a_vis * vis_out).sum(dim=1)
        else:
            # Last-timestep readout (default).
            kin_pooled = kin_out[:, -1, :]
            vis_pooled = vis_out[:, -1, :]
        fused = torch.cat([kin_pooled, vis_pooled], dim=-1)
        return self._head(self._dropout(fused)).squeeze(-1)

    @torch.no_grad()
    def predict_proba(self, x_kin: Tensor, x_cnn: Tensor) -> Tensor:
        self.eval()
        return torch.sigmoid(self.forward(x_kin, x_cnn))
