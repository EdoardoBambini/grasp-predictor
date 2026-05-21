"""PCGrad: Projecting Conflicting Gradients (Yu et al. 2020, NeurIPS).

Wraps a torch optimizer to perform gradient surgery in multi-task settings.
For each pair of task gradients (g_i, g_j), if cos(g_i, g_j) < 0 the algorithm
projects g_i onto the plane normal to g_j:

    g_i <- g_i - (g_i . g_j / ||g_j||^2) * g_j

This prevents a dominant task gradient from cancelling a subordinate one when
they point into opposing semispaces, the failure mode behind the DROID-vs-Fractal
class_separation collapse where DROID's terminal-impulse gradient eats Fractal's
weaker task-success signal.

Standard reduction over the per-task projected gradients: 'mean' (default) or
'sum'. The optimizer's step() is unchanged; only the .grad accumulation is
intercepted.
"""
from __future__ import annotations

import random
from typing import List, Tuple

import numpy as np
import torch
from torch import Tensor


class PCGrad:
    def __init__(self, optimizer, reduction: str = "mean") -> None:
        if reduction not in ("mean", "sum"):
            raise ValueError(f"reduction must be mean or sum, got {reduction!r}")
        self._optim = optimizer
        self._reduction = reduction

    @property
    def optimizer(self):
        return self._optim

    @property
    def param_groups(self):
        return self._optim.param_groups

    def state_dict(self):
        return self._optim.state_dict()

    def load_state_dict(self, sd):
        self._optim.load_state_dict(sd)

    def zero_grad(self, set_to_none: bool = True) -> None:
        self._optim.zero_grad(set_to_none=set_to_none)

    def step(self) -> None:
        self._optim.step()

    def pc_backward(self, losses: List[Tensor]) -> None:
        """Replaces loss.backward() + grad accumulation for multi-task batches.

        Each loss in `losses` must be a scalar tensor with grad_fn (i.e. came
        from criterion(logits[mask_i], y[mask_i])). The method:
          1. Backprops each loss independently and snapshots the flat gradient.
          2. For every pair (i, j) in randomised order, projects g_i off g_j
             when cos(g_i, g_j) < 0.
          3. Reduces the projected gradients and writes them back into .grad
             on the wrapped optimiser's parameter groups.
        Caller is responsible for calling optimizer.step() afterwards.
        """
        if len(losses) == 0:
            return
        if len(losses) == 1:
            losses[0].backward()
            return
        grads, shapes, has_grads = self._pack_grad(losses)
        proj = self._project_conflicting(grads, has_grads)
        unflat = self._unflatten_grad(proj, shapes[0])
        self._set_grad(unflat)

    def _project_conflicting(
        self, grads: List[Tensor], has_grads: List[Tensor],
    ) -> Tensor:
        shared = torch.stack(has_grads).prod(0).bool()
        pc_grad = [g.clone() for g in grads]
        for g_i in pc_grad:
            order = list(range(len(grads)))
            random.shuffle(order)
            for k in order:
                g_j = grads[k]
                dot = torch.dot(g_i, g_j)
                if dot < 0:
                    g_i.sub_(dot * g_j / (g_j.norm() ** 2 + 1e-12))
        stacked = torch.stack(pc_grad)
        merged = torch.zeros_like(grads[0])
        if self._reduction == "mean":
            merged[shared] = stacked[:, shared].mean(dim=0)
        else:
            merged[shared] = stacked[:, shared].sum(dim=0)
        merged[~shared] = stacked[:, ~shared].sum(dim=0)
        return merged

    def _pack_grad(
        self, losses: List[Tensor],
    ) -> Tuple[List[Tensor], List[List[torch.Size]], List[Tensor]]:
        grads, shapes_per_loss, has_grads = [], [], []
        for k, loss in enumerate(losses):
            self._optim.zero_grad(set_to_none=True)
            loss.backward(retain_graph=(k < len(losses) - 1))
            g_list, shape_list, h_list = self._retrieve_grad()
            grads.append(self._flatten(g_list))
            has_grads.append(self._flatten(h_list))
            shapes_per_loss.append(shape_list)
        return grads, shapes_per_loss, has_grads

    def _retrieve_grad(self):
        g_list, shape_list, h_list = [], [], []
        for group in self._optim.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    g_list.append(torch.zeros_like(p))
                    h_list.append(torch.zeros_like(p))
                else:
                    g_list.append(p.grad.detach().clone())
                    h_list.append(torch.ones_like(p))
                shape_list.append(p.shape)
        return g_list, shape_list, h_list

    @staticmethod
    def _flatten(tensors: List[Tensor]) -> Tensor:
        return torch.cat([t.flatten() for t in tensors])

    @staticmethod
    def _unflatten_grad(flat: Tensor, shapes: List[torch.Size]) -> List[Tensor]:
        out, idx = [], 0
        for shape in shapes:
            n = int(np.prod(shape))
            out.append(flat[idx:idx + n].view(shape).clone())
            idx += n
        return out

    def _set_grad(self, grads: List[Tensor]) -> None:
        idx = 0
        for group in self._optim.param_groups:
            for p in group["params"]:
                p.grad = grads[idx]
                idx += 1
