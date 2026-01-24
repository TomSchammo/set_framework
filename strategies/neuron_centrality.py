import numpy as np
from typing import Tuple, Optional, Dict, Any

from .base_strategy import BaseSETStrategy


class NeuronCentralitySET(BaseSETStrategy):
    """
    Buffer / in-place NeuronCentrality SET.

    prune_neurons: in-place prune smallest-magnitude fraction among ACTIVE edges.
    regrow_neurons: in-place regrow using neuron-importance bias (source+target).
                    Supports layers: layer_1, layer_2, layer_3, skip_02
    """

    def __init__(self, zeta=0.3, alpha=0.2, eps=1e-12, seed=None):
        super().__init__(zeta=float(zeta))
        self.alpha = float(alpha)
        self.eps = float(eps)
        self.rng = np.random.default_rng(seed)

        
    def input_layer_importance(self, W, mask=None):
        # importance per input neuron (rows)
        A = np.abs(W)
        if mask is not None:
            A *= mask.astype(bool)
        I = A.sum(axis=1)
        I = I / (I.mean() + self.eps)
        return I

    def hidden_layer_neuron_importance(self, W_in, M_in=None, W_out=None, M_out=None):
        # neuron importance for "current" layer = columns of W_in (and rows of W_out)
        A_in = np.abs(W_in)
        if M_in is not None:
            A_in *= M_in.astype(bool)

        incoming = A_in.sum(axis=0)  # per target neuron (cols)

        if W_out is None:
            I = incoming.copy()
        else:
            A_out = np.abs(W_out)
            if M_out is not None:
                A_out *= M_out.astype(bool)
            outgoing = A_out.sum(axis=1)  # per source neuron of W_out (rows)
            I = incoming * outgoing

        I = np.clip(I.astype(np.float64), 0.0, None)
        I = np.log1p(I)
        I = I / (I.mean() + self.eps)
        return I

    
    def prune_neurons(
        self,
        mask_buffer: np.ndarray,
        weight_values: np.ndarray,
        weight_positions: Optional[np.ndarray] = None,
        extra_info: Optional[Dict[str, Any]] = None,
    ) -> np.ndarray:
        """
        mask_buffer is 2D (same shape as weights matrix).
        weight_values is FLAT (weights.ravel()).
        weight_positions is FLAT (mask.ravel()).
        """
        mb = mask_buffer.ravel()
        w = np.asarray(weight_values).ravel()
        if weight_positions is None:
            m = (mb != 0)
        else:
            m = np.asarray(weight_positions).ravel().astype(bool)

        if mb.shape != w.shape or mb.shape != m.shape:
            raise ValueError(f"Shape mismatch: mb{mb.shape}, w{w.shape}, m{m.shape}")

        # start from current mask
        np.copyto(mb, m.astype(mb.dtype, copy=False))

        active = np.flatnonzero(m)
        n_active = active.size
        if n_active == 0:
            mb[:] = 0
            return mask_buffer

        k = int(self.zeta * n_active)
        if k <= 0:
            return mask_buffer

        # prune smallest |w| among active
        vals = np.abs(w[active])
        order = np.argsort(vals)  # ascending
        prune_idx = active[order[:k]]
        mb[prune_idx] = 0
        return mask_buffer

    def regrow_neurons(
        self,
        num_to_add: int,
        dimensions: Tuple[int, int],
        mask: np.ndarray,
        extra_info: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        In-place regrow: sets mask[i,j]=1 for new edges.
        Uses importance-biased sampling of rows+cols (no argwhere scan).
        """
        k = int(num_to_add)
        if k <= 0:
            return

        if extra_info is None or "self" not in extra_info or "layer" not in extra_info:
            # fallback: uniform
            self._regrow_uniform(k, dimensions, mask)
            return

        sf = extra_info["self"]
        layer = extra_info["layer"]

        n_rows, n_cols = dimensions

        # compute I_source (rows) and I_target (cols) depending on layer
        if layer == "layer_1":
            I_source = self.input_layer_importance(sf.w1[0], sf.wm1_buffer)              # (3072,)
            I_target = self.hidden_layer_neuron_importance(sf.w1[0], sf.wm1_buffer,
                                                          sf.w2[0], sf.wm2_buffer)      # (4000,)
        elif layer == "layer_2":
            I_source = self.hidden_layer_neuron_importance(sf.w1[0], sf.wm1_buffer,
                                                           sf.w2[0], sf.wm2_buffer)     # (4000,)
            I_target = self.hidden_layer_neuron_importance(sf.w2[0], sf.wm2_buffer,
                                                           sf.w3[0], sf.wm3_buffer)     # (1000,)
        elif layer == "layer_3":
            I_source = self.hidden_layer_neuron_importance(sf.w2[0], sf.wm2_buffer,
                                                           sf.w3[0], sf.wm3_buffer)     # (1000,)
            I_target = self.hidden_layer_neuron_importance(sf.w3[0], sf.wm3_buffer,
                                                           sf.w4[0], None)              # (4000,)
        elif layer == "skip_02":
            # skip matrix shape: (3072, 1000)
            I_source = self.input_layer_importance(sf.wSkip02[0], sf.wmSkip02_buffer)   # (3072,)
            I_target = self.hidden_layer_neuron_importance(sf.w2[0], sf.wm2_buffer,
                                                           sf.w3[0], sf.wm3_buffer)     # (1000,)
        else:
            self._regrow_uniform(k, dimensions, mask)
            return

        # sanity sizes
        if I_source.size != n_rows or I_target.size != n_cols:
            self._regrow_uniform(k, dimensions, mask)
            return

        # probs
        ps = np.clip(I_source.astype(np.float64), 0.0, None) + self.eps
        pt = np.clip(I_target.astype(np.float64), 0.0, None) + self.eps
        ps = ps / ps.sum()
        pt = pt / pt.sum()

        added = 0
        attempts = 0
        max_attempts = max(10_000, 50 * k)

        while added < k and attempts < max_attempts:
            attempts += 1

            # sample row/col with importance bias
            i = int(self.rng.choice(n_rows, p=ps))
            j = int(self.rng.choice(n_cols, p=pt))

            if mask[i, j] == 0:
                mask[i, j] = 1
                added += 1

        if added < k:
            # fallback fill uniformly if we couldn't add enough
            self._regrow_uniform(k - added, dimensions, mask)

    def _regrow_uniform(self, k: int, dimensions: Tuple[int, int], mask: np.ndarray) -> None:
        n_rows, n_cols = dimensions
        added = 0
        attempts = 0
        max_attempts = max(10_000, 50 * k)
        while added < k and attempts < max_attempts:
            attempts += 1
            i = int(self.rng.integers(0, n_rows))
            j = int(self.rng.integers(0, n_cols))
            if mask[i, j] == 0:
                mask[i, j] = 1
                added += 1
