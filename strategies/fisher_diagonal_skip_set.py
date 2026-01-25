import numpy as np
from strategies.base_strategy import BaseSETStrategy

class FisherDiagonalSkipSET(BaseSETStrategy):
    """
    Fisher diagonal strategy compatible with NEW buffer-style SET:
    - prune_neurons writes into mask_buffer (in-place)
    - regrow_neurons writes into mask_buffer (in-place)

    Prune score (keep-utility):   w^2 / (v + eps)    (keep highest, prune lowest zeta)
    Grow score (for inactive):    g^2 / (v + eps)    (activate best candidates)
    """

    needs_gradients = True

    def __init__(self, zeta=0.3, beta=0.9, eps=1e-8, seed=None):
        self.zeta = float(zeta)
        self.beta = float(beta)
        self.eps = float(eps)
        self.rng = np.random.default_rng(seed)

        # filled by set_keras during training
        self.v1 = None
        self.v2 = None
        self.v3 = None
        self.vSkip02 = None

    def prune_neurons(self, mask_buffer: np.ndarray, weights_flat: np.ndarray, mask_flat: np.ndarray, extra_info=None):
        """
        In-place prune:
          mask_buffer is a FLAT OR SHAPED buffer (your caller uses shaped).
          We'll write a shaped mask into it.
        """
        if extra_info is None or "v_flat" not in extra_info:
            raise ValueError("FisherDiagonalSkipSET.prune_neurons requires extra_info['v_flat'].")

        w = weights_flat.astype(np.float32, copy=False)
        m = mask_flat.astype(np.float32, copy=False)
        v = extra_info["v_flat"].astype(np.float32, copy=False)

        active = np.flatnonzero(m > 0.0)
        if active.size == 0:
            # keep as-is
            np.copyto(mask_buffer, m.reshape(mask_buffer.shape))
            return mask_buffer

        n_active = active.size
        n_keep = int(np.round((1.0 - self.zeta) * n_active))
        n_keep = max(0, min(n_active, n_keep))

        score = (w[active] * w[active]) / (v[active] + self.eps)

        # build new keep mask flat
        keep = np.zeros_like(m, dtype=np.float32)
        if n_keep > 0:
            order = np.argsort(score)          # low -> high
            keep_idx = active[order[-n_keep:]] # take top
            keep[keep_idx] = 1.0

        # write into mask_buffer in the SAME SHAPE the caller expects
        np.copyto(mask_buffer, keep.reshape(mask_buffer.shape))
        return mask_buffer

    def regrow_neurons(self, noRewires: int, shape, mask_buffer: np.ndarray, extra_info=None):
        """
        In-place regrow: sets mask_buffer[i,j]=1 for chosen inactive edges.
        """

        assert noRewires > 0, "Expected at least one wire"


        rows, cols = shape
        active_bool = (mask_buffer > 0.0)
        inactive = np.argwhere(~active_bool)
        
        if inactive.shape[0] == 0:
            assert ValueError("Inactive shape is zero; nothing to regrow!")

        # If no extra_info, value error
        if extra_info is None or ("g" not in extra_info) or ("v" not in extra_info):
            # self._regrow_random_inplace(noRewires, rows, cols, mask_buffer)
            assert ValueError("FisherDiagonalSkipSET.regrow_neurons requires extra_info['g'] and extra_info['v'].")

        g = extra_info["g"]
        v = extra_info["v"]

        cand = min(inactive.shape[0], noRewires * 20)
        idx = self.rng.choice(inactive.shape[0], size=cand, replace=False)
        c = inactive[idx]
        ci, cj = c[:, 0], c[:, 1]

        score = (g[ci, cj] * g[ci, cj]) / (v[ci, cj] + self.eps)
        order = np.argsort(score)[::-1]  # high -> low

        grown = 0
        for k in order:
            i = int(ci[k]); j = int(cj[k])
            if mask_buffer[i, j] != 0.0:
                continue
            mask_buffer[i, j] = 1.0
            grown += 1
            if grown >= noRewires:
                return

        # fallback if not enough
        if grown < noRewires:
            self._regrow_random_inplace(noRewires - grown, rows, cols, mask_buffer)

    def _regrow_random_inplace(self, k, rows, cols, mask_buffer):
        tries = 0
        max_tries = k * 500
        grown = 0
        while grown < k and tries < max_tries:
            i = int(self.rng.integers(0, rows))
            j = int(self.rng.integers(0, cols))
            tries += 1
            if mask_buffer[i, j] != 0.0:
                continue
            mask_buffer[i, j] = 1.0
            grown += 1
