import numpy as np
from typing import Tuple, Optional, Set, List

from .base_strategy import BaseSETStrategy


class NeuronCentralitySET(BaseSETStrategy):

    def __init__(self, zeta=0.3, alpha=0.7, eps=1e-12):
        super().__init__(zeta)
        self.alpha = float(alpha)
        self.eps = float(eps)

    def input_layer_importance(self, W1, mask=None, eps=1e-12):
        A = np.abs(W1)
        if mask is not None:
            A *= mask.astype(bool)
        I0 = A.sum(axis=1)
        I0 = I0 / (I0.mean() + eps)
        return I0
    
    def hidden_layer_neuron_importance(
        self,
        W_in,  M_in=None,     # (n_cur,  n_prev)
        W_out=None, M_out=None,  # (n_next, n_cur)
        eps=1e-12,
    ):
        W = W_in
        A_in = np.abs(W)
        if M_in is not None:
            A_in *= M_in.astype(bool)

        incoming = A_in.sum(axis=0)  # sum over prev neurons

        if W_out is None:
            # If no outgoing matrix provided
            I = incoming.copy()
        else:
            A_out = np.abs(W_out)
            if M_out is not None:
                A_out *= M_out.astype(bool)

            outgoing = A_out.sum(axis=1)  # sum over next neurons
            I = incoming * outgoing

        I = np.clip(I.astype(np.float64), 0.0, None)
        I = np.log1p(I) # Taming the monsters :)
        I = I / (I.mean() + eps)

        return I
    
    def prune_neurons(self, weight_values, weight_positions=None, extra_info=None):

        w = np.asarray(weight_values).ravel()

        if weight_positions is None:
            m = (w != 0)  # fallback
        else:
            m = np.asarray(weight_positions).ravel().astype(bool)

        # shapes must match
        if m.shape != w.shape:
            raise ValueError(f"Mask shape {m.shape} != weight shape {w.shape}")

        nz_idx = np.where(m)[0]
        n = nz_idx.size

        # Must return an array of same length as w
        keep = m.copy()

        if n == 0:
            return keep  # all False

        # prune smallest-magnitude fraction among existing edges
        k = int(self.zeta * n)
        k = max(1, min(k, n))

        nz_vals = w[nz_idx]
        sorted_local = np.argsort(np.abs(nz_vals))
        prune_idx = nz_idx[sorted_local[:k]]

        keep[prune_idx] = False
        return keep
    
    def regrow_neurons(self, num_to_add, dimensions, existing_positions, extra_info = None):
        n_rows, n_cols = dimensions
        k = int(num_to_add)
        if k <= 0:
            return []

        if extra_info is None:
            return [] # Can't regrow without importance
        
        ex = extra_info

        sf = ex["self"]
        if sf is None:
            return []

        match ex["layer"]:
            case "layer_1":
                I_source = self.input_layer_importance(sf.w1[0], sf.wm1)
                I_target = self.hidden_layer_neuron_importance(W_in=sf.w1[0], M_in=sf.wm1, W_out=sf.w2[0], M_out=sf.wm2)  # (4000,)
            case "layer_2":
                I_source = self.hidden_layer_neuron_importance(W_in=sf.w1[0], M_in=sf.wm1, W_out=sf.w2[0], M_out=sf.wm2)  # (4000,)
                I_target = self.hidden_layer_neuron_importance(W_in=sf.w2[0], M_in=sf.wm2, W_out=sf.w3[0], M_out=sf.wm3)  # (1000,)
            case "layer_3":
                I_source = self.hidden_layer_neuron_importance(W_in=sf.w2[0], M_in=sf.wm2, W_out=sf.w3[0], M_out=sf.wm3)  # (1000,)
                I_target = self.hidden_layer_neuron_importance(W_in=sf.w3[0], M_in=sf.wm3, W_out=sf.w4[0], M_out=None)   # (4000,)

        # Build candidate dead edges
        zeros = [(r, c) for r in range(n_rows) for c in range(n_cols)
                if (r, c) not in existing_positions]

        N_zero = len(zeros)
        if N_zero == 0:
            return []

        k = min(k, N_zero)
        if k <= 0:
            return []

        # If importance vectors are missing or wrong size, fall back to uniform
        probs = None
        if I_target is not None and I_source is not None:
            I_target = np.clip(np.asarray(I_target, dtype=np.float64), 0.0, None)
            I_source = np.clip(np.asarray(I_source, dtype=np.float64), 0.0, None)

            # Your original mapping uses:
            #   i = col index  -> I_target
            #   j = row index  -> I_source
            # So sizes should be: I_target len == n_cols, I_source len == n_rows
            if I_target.size == n_cols and I_source.size == n_rows:
                rows = np.fromiter((p[0] for p in zeros), dtype=np.int64, count=N_zero)
                cols = np.fromiter((p[1] for p in zeros), dtype=np.int64, count=N_zero)

                # Preserve your original behavior:
                imp_prod = I_target[cols] * I_source[rows]

                # (Normal mapping would be: I_target[rows] * I_source[cols])

                scores = self.alpha * imp_prod + (1.0 - self.alpha) * (1.0 / float(N_zero))

                ssum = scores.sum()
                if np.isfinite(ssum) and ssum > self.eps:
                    probs = scores / ssum  # else leave as None for uniform

        chosen = np.random.choice(N_zero, size=k, replace=False, p=probs)
        return [zeros[idx] for idx in chosen]