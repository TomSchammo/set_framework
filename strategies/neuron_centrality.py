import numpy as np
from typing import Tuple, Optional

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
        W_in,
        M_in=None,  # (n_cur,  n_prev)
        W_out=None,
        M_out=None,  # (n_next, n_cur)
        eps=1e-12,
    ):
        # Convention: W_in shape (n_prev, n_cur), W_out shape (n_cur, n_next)
        # incoming: sum over prev (axis=0) -> (n_cur,)
        # outgoing: sum over next (axis=1) -> (n_cur,)
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
        I = np.log1p(I)  # Taming the monsters :)
        I = I / (I.mean() + eps)

        return I

    def prune_neurons(self,
                      mask_buffer: np.ndarray,
                      weight_values: np.ndarray,
                      weight_positions: Optional[np.ndarray] = None,
                      extra_info: Optional[dict] = None) -> np.ndarray:

        w = weight_values

        # NOTE: we can assume this to be the case in our code
        assert weight_positions is not None

        m = weight_positions

        # shapes must match
        if m.shape != w.shape:
            raise ValueError(f"Mask shape {m.shape} != weight shape {w.shape}")

        nz_idx = np.where(m)[0]
        n = nz_idx.size

        mask_buffer_flat = mask_buffer.ravel()

        assert mask_buffer.__array_interface__['data'][
            0] == mask_buffer_flat.__array_interface__['data'][
                0], "Unexpected copy"

        # Must return an array of same length as w
        np.copyto(mask_buffer_flat, m)

        if n == 0:
            return mask_buffer  # all False

        # prune smallest-magnitude fraction among existing edges
        k = int(self.zeta * n)
        k = max(1, min(k, n))

        nz_vals = w[nz_idx]
        sorted_local = np.argsort(np.abs(nz_vals))
        prune_idx = nz_idx[sorted_local[:k]]

        mask_buffer_flat[prune_idx] = 0
        return mask_buffer

    def regrow_neurons(self,
                       num_to_add: int,
                       dimensions: Tuple[int, int],
                       mask: np.ndarray,
                       extra_info: Optional[dict] = None) -> None:
        
        n_rows, n_cols = dimensions
        k = int(num_to_add)

        assert extra_info, "Importance has to be provided for this strategy to work"

        ex = extra_info

        sf = ex["self"]

        if sf is None:
            assert False

        match ex["layer"]:
            case "layer_1":
                I_source = self.input_layer_importance(sf.w1[0], sf.wm1_buffer)
                I_target = self.hidden_layer_neuron_importance(
                    W_in=sf.w1[0],
                    M_in=sf.wm1_buffer,
                    W_out=sf.w2[0],
                    M_out=sf.wm2_buffer)  # (4000,)
            case "layer_2":
                I_source = self.hidden_layer_neuron_importance(
                    W_in=sf.w1[0],
                    M_in=sf.wm1_buffer,
                    W_out=sf.w2[0],
                    M_out=sf.wm2_buffer)  # (4000,)
                I_target = self.hidden_layer_neuron_importance(
                    W_in=sf.w2[0],
                    M_in=sf.wm2_buffer,
                    W_out=sf.w3[0],
                    M_out=sf.wm3_buffer)  # (1000,)
            case "layer_3":
                I_source = self.hidden_layer_neuron_importance(
                    W_in=sf.w2[0],
                    M_in=sf.wm2_buffer,
                    W_out=sf.w3[0],
                    M_out=sf.wm3_buffer)  # (1000,)
                I_target = self.hidden_layer_neuron_importance(
                    W_in=sf.w3[0],
                    M_in=sf.wm3_buffer,
                    W_out=sf.w4[0],
                    M_out=None)  # (4000,)
            case _:
                raise ValueError(f"Invalid layer '{ex['layer']}'")

        # Build candidate dead edges
        zr, zc = np.where(mask == 0)
        N_zero = zr.size

        print(f"Debug : N_zero = {N_zero}, k = {k}")
        assert N_zero > 0, "There should be neurons that are not connected!"

        k = min(k, N_zero)

        # If importance vectors are missing or wrong size, fall back to uniform
        probs = None
        if I_target is not None and I_source is not None:
            I_target = np.clip(np.asarray(I_target, dtype=np.float64), 0.0,
                               None)
            I_source = np.clip(np.asarray(I_source, dtype=np.float64), 0.0,
                               None)

            if I_target.size == n_cols and I_source.size == n_rows:
                # Preserve your original behavior:
                imp_prod = I_target[zc] * I_source[zr]

                #scores = self.alpha * imp_prod + (1.0 - self.alpha) * (1.0 / float(N_zero))
                scores = self.alpha * imp_prod + (1.0 - self.alpha) * (1.0)

                ssum = scores.sum()
                if np.isfinite(ssum) and ssum > self.eps:
                    probs = scores / ssum  # else leave as None for uniform

        chosen = np.random.choice(N_zero, size=k, replace=False, p=probs)
        mask[zr[chosen], zc[chosen]] = 1
