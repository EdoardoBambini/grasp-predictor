"""Multiple Instance Learning model on top of the cached visuo-kinematic features.

Drops the per-window classification head used by LateFusionLSTM and replaces it
with a bag-level pipeline:
  - WindowEncoder produces a 256-dim embedding per window (same recipe as
    LateFusionLSTM, minus the head).
  - GatedAttentionPool (Ilse et al., ICML 2018) reduces the M windows of a bag
    to a single bag embedding via tanh*sigmoid gated attention.
  - A small MLP head outputs a single logit per bag.

The bag = an entire sequence; the windows inside it are the instances. The
loss is computed once per bag against the bag's sequence-level label, which
removes the noisy-supervision signal that hurts DROID and Fractal RT-1 in the
window-level training (every window of a failure episode currently inherits
the "failure" label even when it shows clearly successful approach motion).
"""
from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor


class WindowEncoder(nn.Module):
    """Reuse of LateFusionLSTM's encoder (kin LSTM + vis LSTM + optional
    per-frame attention pool), exposing the fused window embedding instead of
    a classification logit. Output dim = (kin_hidden + vis_hidden) * directions."""

    def __init__(
        self,
        kin_dim: int = 15,
        cnn_dim: int = 64,
        kin_hidden: int = 64,
        vis_hidden: int = 64,
        num_layers: int = 1,
        dropout: float = 0.3,
        bidirectional: bool = True,
        attn_pool: bool = True,
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
        self._attn_pool = attn_pool
        if attn_pool:
            self._attn_kin = nn.Linear(kin_hidden * directions, 1)
            self._attn_vis = nn.Linear(vis_hidden * directions, 1)
        else:
            self._attn_kin = None
            self._attn_vis = None
        self.win_embed_dim = (kin_hidden + vis_hidden) * directions

    def forward(self, x_kin: Tensor, x_cnn: Tensor) -> Tensor:
        """x_kin: (B, T, kin_dim), x_cnn: (B, T, cnn_dim). Returns (B, embed_dim)."""
        kin_out, _ = self._lstm_kin(x_kin)
        vis_out, _ = self._lstm_vis(x_cnn)
        if self._attn_pool:
            a_kin = torch.softmax(self._attn_kin(kin_out), dim=1)
            kin_pooled = (a_kin * kin_out).sum(dim=1)
            a_vis = torch.softmax(self._attn_vis(vis_out), dim=1)
            vis_pooled = (a_vis * vis_out).sum(dim=1)
        else:
            kin_pooled = kin_out[:, -1, :]
            vis_pooled = vis_out[:, -1, :]
        return torch.cat([kin_pooled, vis_pooled], dim=-1)


class GatedAttentionPool(nn.Module):
    """Gated attention pooling over a bag of instance embeddings.

    Ilse et al., "Attention-based Deep Multiple Instance Learning", ICML 2018.

      a_i = w^T (tanh(V h_i) * sigmoid(U h_i))
      a   = softmax(a_i over valid i)
      z   = sum_i a_i h_i
    """

    def __init__(self, embed_dim: int, hidden: int = 128) -> None:
        super().__init__()
        self._V = nn.Linear(embed_dim, hidden, bias=False)
        self._U = nn.Linear(embed_dim, hidden, bias=False)
        self._w = nn.Linear(hidden, 1, bias=False)

    def forward(self, H: Tensor, mask: Tensor | None = None) -> tuple[Tensor, Tensor]:
        """
        H:    (B, M, embed_dim) - instance embeddings (M = max windows in batch)
        mask: (B, M) bool - True where instance is valid (not padding)

        Returns:
          bag:  (B, embed_dim) bag embedding
          attn: (B, M) attention weights (softmax-normalized over valid)
        """
        v = torch.tanh(self._V(H))
        u = torch.sigmoid(self._U(H))
        logits = self._w(v * u).squeeze(-1)  # (B, M)
        if mask is not None:
            logits = logits.masked_fill(~mask, float("-inf"))
        attn = torch.softmax(logits, dim=1)
        bag = (attn.unsqueeze(-1) * H).sum(dim=1)
        return bag, attn


class LateFusionLSTMMIL(nn.Module):
    """End-to-end MIL classifier: WindowEncoder -> pool -> MLP head.

    pool_mode selects the bag-level aggregator:
      - "gated": GatedAttentionPool (Ilse 2018). Softmax over all valid
        instances -> can over-smooth when the failure signal is impulsive.
      - "max":   masked max-pool over instances. Routes gradient through the
        single most-activated window per dim. Used by v10g to break the
        attention-collapse failure mode of v10f.
      - "mean":  masked mean-pool over instances. Baseline; equivalent to a
        fully-degenerated attention.

    Forward signature is bag-level:
      X_kin: (B, M, T, kin_dim)
      X_cnn: (B, M, T, cnn_dim)
      mask:  (B, M) bool  - True where window is valid; default all-True

    Returns:
      logits: (B,)
      attn:   (B, M) - gated weights for "gated"; argmax one-hot for "max";
              uniform-over-valid for "mean". Kept for logging compatibility.
    """

    def __init__(
        self,
        kin_dim: int = 15,
        cnn_dim: int = 64,
        kin_hidden: int = 64,
        vis_hidden: int = 64,
        num_layers: int = 1,
        dropout: float = 0.3,
        bidirectional: bool = True,
        attn_pool: bool = True,
        mil_attn_hidden: int = 128,
        head_hidden: int = 128,
        pool_mode: str = "gated",
        use_film: bool = False,
        nl_emb_dim: int = 512,
        film_hidden: int = 64,
        use_nl_concat: bool = False,
        nl_concat_dim: int = 64,
    ) -> None:
        super().__init__()
        if pool_mode not in ("gated", "max", "mean"):
            raise ValueError(
                f"pool_mode must be one of gated/max/mean, got {pool_mode!r}",
            )
        if use_film and use_nl_concat:
            raise ValueError(
                "use_film and use_nl_concat are mutually exclusive: pick one "
                "language-conditioning path (FiLM post-pool OR concat pre-pool).",
            )
        self._pool_mode = pool_mode
        self._use_film = bool(use_film)
        self._use_nl_concat = bool(use_nl_concat)
        self._nl_concat_dim = int(nl_concat_dim) if use_nl_concat else 0
        self.encoder = WindowEncoder(
            kin_dim=kin_dim, cnn_dim=cnn_dim,
            kin_hidden=kin_hidden, vis_hidden=vis_hidden,
            num_layers=num_layers, dropout=dropout,
            bidirectional=bidirectional, attn_pool=attn_pool,
        )
        # When NL concat is on, instances entering the bag pool carry the
        # projected language vector as extra channels. The pool and head both
        # widen by nl_concat_dim. This bypasses the identity-init trap of FiLM
        # (v10k) by routing language through a Xavier-initialised Linear with
        # immediate non-zero gradient flow from step 0.
        pool_in_dim = self.encoder.win_embed_dim + self._nl_concat_dim
        if pool_mode == "gated":
            self.bag_pool = GatedAttentionPool(
                embed_dim=pool_in_dim,
                hidden=mil_attn_hidden,
            )
        else:
            self.bag_pool = None
        self.dropout = nn.Dropout(p=dropout)
        self.head = nn.Sequential(
            nn.Linear(pool_in_dim, head_hidden),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(head_hidden, 1),
        )

        # v10k: FiLM modulation of the bag-level embedding from a per-bag
        # natural-language embedding (Universal Sentence Encoder, 512-D pulled
        # from the Fractal TFRecord). Brohan 2022 (RT-1) / Jang 2021 (BC-Z)
        # idiom. Identity-initialised: the second Linear has weights and bias
        # set to zero, so gamma=0 and beta=0 at step 0 -> (1+gamma)*bag_emb +
        # beta = bag_emb (no-op). The model is free to learn task-conditional
        # modulation only when it helps reduce loss.
        #
        # DROID samples use a learnable `nl_null` (512-D) instead of the
        # Fractal NL embedding; routing happens in forward() via a per-bag
        # `is_droid` mask. With nl_null initialised to zeros AND film weights
        # set to zero, DROID training starts numerically identical to the
        # no-FiLM model, protecting the existing DROID class_sep = 0.70.
        if self._use_film:
            d_bag = self.encoder.win_embed_dim
            self.film_gen = nn.Sequential(
                nn.Linear(nl_emb_dim, film_hidden),
                nn.GELU(),
                nn.Linear(film_hidden, 2 * d_bag),
            )
            # Identity init on the output layer (zero weights + zero bias)
            with torch.no_grad():
                self.film_gen[-1].weight.zero_()
                self.film_gen[-1].bias.zero_()
            # Null token for DROID samples
            self.nl_null = nn.Parameter(torch.zeros(nl_emb_dim))
        else:
            self.film_gen = None
            self.nl_null = None

        # v10l: pre-attention NL concat. Project the 512-D USE embedding down to
        # nl_concat_dim and append it to every instance embedding before the
        # Gated Attention pool. Xavier-init (default for nn.Linear) so gradient
        # flows from step 0 - no identity-init trap. DROID samples are routed
        # through a learnable nl_null parameter (zero-init) via the per-bag
        # is_droid mask in forward(), so DROID training is numerically close to
        # v10h at step 0 (nl_proj(zeros) = nl_proj.bias, small and learnable).
        if self._use_nl_concat:
            self.nl_proj = nn.Linear(nl_emb_dim, self._nl_concat_dim)
            self.nl_null = nn.Parameter(torch.zeros(nl_emb_dim))
        else:
            self.nl_proj = None

    def forward(
        self,
        X_kin: Tensor,
        X_cnn: Tensor,
        mask: Tensor | None = None,
        nl_emb: Tensor | None = None,
        is_droid: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        B, M, T, _ = X_kin.shape
        # Encode every window in a single LSTM call by flattening bag axis.
        win_emb = self.encoder(
            X_kin.reshape(B * M, T, -1),
            X_cnn.reshape(B * M, T, -1),
        )  # (B*M, embed_dim)
        win_emb = win_emb.view(B, M, -1)

        # v10l: pre-attention NL concat. Route per-bag NL emb (Fractal) or
        # learnable null (DROID), project, broadcast over instances, append.
        if self._use_nl_concat and self.nl_proj is not None:
            if nl_emb is None:
                nl_in = self.nl_null.expand(B, -1)
            elif is_droid is not None:
                null_expanded = self.nl_null.expand(B, -1)
                nl_in = torch.where(is_droid.view(B, 1), null_expanded, nl_emb)
            else:
                nl_in = nl_emb
            nl_p = self.nl_proj(nl_in)              # (B, nl_concat_dim)
            nl_p = nl_p.unsqueeze(1).expand(B, M, -1)  # (B, M, nl_concat_dim)
            win_emb = torch.cat([win_emb, nl_p], dim=-1)

        bag_emb, attn = self._pool_bag(win_emb, mask=mask)

        # v10k: FiLM modulation on the bag embedding using the per-bag NL
        # embedding (Fractal) or learnable null token (DROID).
        if self._use_film and self.film_gen is not None:
            if nl_emb is None:
                # Inference fallback: assume null routing (DROID-like)
                nl_in = self.nl_null.expand(B, -1)
            else:
                if is_droid is not None:
                    # Per-sample route: DROID rows take the null token
                    null_expanded = self.nl_null.expand(B, -1)
                    is_droid_bcast = is_droid.view(B, 1)
                    nl_in = torch.where(is_droid_bcast, null_expanded, nl_emb)
                else:
                    nl_in = nl_emb
            gb = self.film_gen(nl_in)  # (B, 2*d_bag)
            d_bag = bag_emb.shape[-1]
            gamma = gb[:, :d_bag]
            beta = gb[:, d_bag:]
            bag_emb = (1.0 + gamma) * bag_emb + beta

        logits = self.head(self.dropout(bag_emb)).squeeze(-1)
        return logits, attn

    def _pool_bag(
        self,
        win_emb: Tensor,
        mask: Tensor | None,
    ) -> tuple[Tensor, Tensor]:
        if self._pool_mode == "gated":
            return self.bag_pool(win_emb, mask=mask)
        B, M, D = win_emb.shape
        if mask is None:
            mask = torch.ones((B, M), dtype=torch.bool, device=win_emb.device)
        mask_f = mask.float()
        if self._pool_mode == "max":
            neg_inf = torch.finfo(win_emb.dtype).min
            masked = win_emb.masked_fill(~mask.unsqueeze(-1), neg_inf)
            bag_emb, max_idx = masked.max(dim=1)  # (B, D), (B, D)
            # Logging proxy: fraction of dims selecting each instance.
            attn = torch.zeros((B, M), dtype=win_emb.dtype, device=win_emb.device)
            ones = torch.ones_like(max_idx, dtype=win_emb.dtype)
            attn.scatter_add_(1, max_idx, ones)
            attn = attn / float(D)
            return bag_emb, attn
        # mean
        denom = mask_f.sum(dim=1, keepdim=True).clamp(min=1.0)
        bag_emb = (win_emb * mask_f.unsqueeze(-1)).sum(dim=1) / denom
        attn = mask_f / denom
        return bag_emb, attn

    @torch.no_grad()
    def predict_proba(
        self,
        X_kin: Tensor,
        X_cnn: Tensor,
        mask: Tensor | None = None,
        nl_emb: Tensor | None = None,
        is_droid: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        self.eval()
        logits, attn = self.forward(X_kin, X_cnn, mask, nl_emb=nl_emb, is_droid=is_droid)
        return torch.sigmoid(logits), attn
