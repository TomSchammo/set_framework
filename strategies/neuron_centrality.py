import numpy as np
from typing import Tuple, Optional, Dict, Any
from .base_strategy import BaseSETStrategy


class NeuronCentralitySET(BaseSETStrategy):
    """
    Buffer/in-place Centrality SET:
      - prune: remove smallest |w| among active edges (same as Random pruning)
      - regrow: biased sampling using neuron-importance

    Set use_skip=True to also rewire the skip_02 matrix (input -> layer2).
    """

    def __init__(self, zeta=0.3, alpha=0.2, eps=1e-12, seed=None, use_skip: bool = False):
        super().__init__(zeta)
        self.alpha = float(alpha)
        self.eps = float(eps)
        self.rng = np.random.default_rng(seed)
        self.uses_skip = bool(use_skip)

    def input_layer_importance(self, W1, mask=None, eps=1e-12):
        A = np.abs(W1)
        if mask is not None:
            A *= mask.astype(bool)
        I0 = A.sum(axis=1)
        I0 = I0 / (I0.mean() + eps)
        return I0

    def hidden_layer_neuron_importance(
        self,
        W_in,
        M_in=None,
        W_out=None,
        M_out=None,
        eps=1e-12,
    ):
        A_in = np.abs(W_in)
        if M_in is not None:
            A_in *= M_in.astype(bool)

        incoming = A_in.sum(axis=0)  # per target neuron (columns)

        if W_out is None:
            I = incoming.copy()
        else:
            A_out = np.abs(W_out)
            if M_out is not None:
                A_out *= M_out.astype(bool)

            outgoing = A_out.sum(axis=1)  # per source neuron of next layer
            I = incoming * outgoing

        I = np.clip(I.astype(np.float64), 0.0, None)
        I = np.log1p(I)
        I = I / (I.mean() + eps)
        return I

    def prune_neurons(
        self,
        mask_buffer: np.ndarray,
        weight_values: np.ndarray,
        weight_positions: Optional[np.ndarray] = None,
        extra_info: Optional[Dict[str, Any]] = None,
    ) -> np.ndarray:
        mb = mask_buffer.ravel()
        w = np.asarray(weight_values).ravel()

        assert weight_positions is not None, "Centrality expects weight_positions (mask.ravel())"
        m = np.asarray(weight_positions).ravel().astype(bool)

        if mb.shape != w.shape or m.shape != w.shape:
            raise ValueError(f"Shape mismatch: mb{mb.shape} w{w.shape} m{m.shape}")

        np.copyto(mb, m.astype(mb.dtype, copy=False))

        active = np.flatnonzero(m)
        n_active = active.size
        if n_active == 0:
            mb[:] = 0
            return mask_buffer

        k = int(self.zeta * n_active)
        k = max(1, min(k, n_active))

        order = np.argsort(np.abs(w[active]))
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
        k = int(num_to_add)
        if k <= 0:
            return

        if not extra_info or "self" not in extra_info or "layer" not in extra_info:
            # fallback uniform
            inactive = np.argwhere(mask == 0)
            if inactive.shape[0] == 0:
                return
            k = min(k, inactive.shape[0])
            pick = self.rng.choice(inactive.shape[0], size=k, replace=False)
            chosen = inactive[pick]
            mask[chosen[:, 0], chosen[:, 1]] = 1
            return

        sf = extra_info["self"]
        layer = extra_info["layer"]

        n_rows, n_cols = dimensions

        
        if layer == "layer_1":
            I_source = self.input_layer_importance(sf.w1[0], sf.wm1_buffer)  # input units
            I_target = self.hidden_layer_neuron_importance(sf.w1[0], sf.wm1_buffer, sf.w2[0], sf.wm2_buffer)  # 4000
        elif layer == "layer_2":
            I_source = self.hidden_layer_neuron_importance(sf.w1[0], sf.wm1_buffer, sf.w2[0], sf.wm2_buffer)  # 4000
            I_target = self.hidden_layer_neuron_importance(sf.w2[0], sf.wm2_buffer, sf.w3[0], sf.wm3_buffer)  # 1000
        elif layer == "layer_3":
            I_source = self.hidden_layer_neuron_importance(sf.w2[0], sf.wm2_buffer, sf.w3[0], sf.wm3_buffer)  # 1000
            I_target = self.hidden_layer_neuron_importance(sf.w3[0], sf.wm3_buffer, sf.w4[0], None)  # 4000
        elif layer == "skip_02":
            # skip connects input -> layer2 (1000)
            I_source = self.input_layer_importance(sf.wSkip02[0], sf.wmSkip02_buffer)
            I_target = self.hidden_layer_neuron_importance(sf.wSkip02[0], sf.wmSkip02_buffer, sf.w3[0], sf.wm3_buffer)
        else:
            raise ValueError(f"Invalid layer '{layer}'")

        inactive = np.argwhere(mask == 0)
        N = inactive.shape[0]
        if N == 0:
            return
        k = min(k, N)

        rows = inactive[:, 0]
        cols = inactive[:, 1]

        
        if I_target.size == n_cols and I_source.size == n_rows:
            imp_prod = I_target[cols] * I_source[rows]
            scores = self.alpha * imp_prod + (1.0 - self.alpha) * 1.0
            ssum = float(scores.sum())
            if np.isfinite(ssum) and ssum > self.eps:
                probs = scores / ssum
            else:
                probs = None
        else:
            probs = None

        pick = self.rng.choice(N, size=k, replace=False, p=probs)
        chosen = inactive[pick]
        mask[chosen[:, 0], chosen[:, 1]] = 1
