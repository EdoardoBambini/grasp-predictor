"""
GraspIntegrityLSTM: binary grasp-failure classifier. Two backbones — tsai
LSTMPlus if installed (stacked LSTM + attention + FC head, Apache 2.0, see
https://github.com/timeseriesAI/tsai), else plain nn.LSTM. The released
checkpoint was trained on the vanilla path.

In (batch, seq_len, n_features) float32; out (batch,) raw logits — pair
with BCEWithLogitsLoss. For inference call predict_proba(x) -> P(fail).
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor


class GraspIntegrityLSTM(nn.Module):
    """Binary grasp-failure classifier. Vanilla path = stacked LSTM + FC head."""

    def __init__(
        self,
        input_size: int,
        hidden_size: int = 128,
        num_layers: int = 2,
        dropout: float = 0.3,
        bidirectional: bool = False,
        fc_dropout: float = 0.1,
    ) -> None:
        super().__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # tsai path: LSTMPlus wants (batch, n_vars, seq_len); input is permuted at forward time.
        try:
            from tsai.models.RNN import LSTMPlus as _LSTMPlus
            self._backbone = _LSTMPlus(
                c_in=input_size,
                c_out=1,               # 1 logit, binary classifier
                hidden_size=hidden_size,
                n_layers=num_layers,
                bias=True,
                rnn_dropout=dropout,
                bidirectional=bidirectional,
                fc_dropout=fc_dropout,
            )
            self._use_tsai = True
        except ImportError:
            # no tsai -> vanilla nn.LSTM. The final checkpoint was trained here.
            import warnings
            warnings.warn(
                "tsai not installed, falling back to vanilla nn.LSTM. "
                "Install tsai via `poetry install` to enable the LSTMPlus path.",
                ImportWarning,
                stacklevel=2,
            )
            self._use_tsai = False
            _directions = 2 if bidirectional else 1
            self._lstm = nn.LSTM(
                input_size=input_size,
                hidden_size=hidden_size,
                num_layers=num_layers,
                dropout=dropout if num_layers > 1 else 0.0,
                bidirectional=bidirectional,
                batch_first=True,
            )
            self._dropout = nn.Dropout(p=dropout)
            self._head = nn.Linear(hidden_size * _directions, 1)

    def forward(self, x: Tensor) -> Tensor:
        # x: (batch, seq_len, n_features) -> logits (batch,). Apply sigmoid for P(fail).
        if self._use_tsai:
            # tsai wants (batch, n_vars, seq_len); input is (batch, seq_len, n_vars).
            x_tsai = x.permute(0, 2, 1)
            out = self._backbone(x_tsai)           # (batch, 1)
            return out.squeeze(-1)
        # vanilla: last timestep -> dropout -> linear.
        lstm_out, _ = self._lstm(x)
        last = lstm_out[:, -1, :]
        return self._head(self._dropout(last)).squeeze(-1)

    @torch.no_grad()
    def predict_proba(self, x: Tensor) -> Tensor:
        # failure probability in [0, 1]. Close to 1 = likely failure.
        self.eval()
        return torch.sigmoid(self.forward(x))
