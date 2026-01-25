import numpy as np
from strategies.base_strategy import BaseSETStrategy

class FisherDiagonalSET(BaseSETStrategy):
    """
    Fisher diagonal (NO SKIP):
      prune: w^2/(v+eps)  keep highest (prune lowest zeta)
      grow:  g^2/(v+eps)  activate best inactive candidates
    """
    needs_gradients = True

    def __init__(self, zeta=0.3, beta=0.9, eps=1e-8, seed=None):
        self.zeta = float(zeta)
        self.beta = float(beta)
        self.eps = float(eps)
        self.rng = np.random.default_rng(seed)

        self.v1 = None
        self.v2 = None
        self.v3 = None

    # buffer-style API (works with the adapter rewireMask you already added)
    def prune_neurons(self, mask_buffer, weights_flat, mask_flat, extra_info=None):
        if extra_info is None or "v_flat" not in extra_info:
            raise ValueError("FisherDiagonalSET needs extra_info['v_flat'].")

        w = weights_flat.astype(np.float32, copy=False)
        m = mask_flat.astype(np.float32, copy=False)
        v = extra_info["v_flat"].astype(np.float32, copy=False)

        active = np.flatnonzero(m > 0.0)
        if active.size == 0:
            mask_buffer[:] = m.reshape(mask_buffer.shape)
            return mask_buffer

        n_active = active.size
        n_keep = int(np.round((1.0 - self.zeta) * n_active))
        n_keep = max(0, min(n_active, n_keep))

        score = (w[active] * w[active]) / (v[active] + self.eps)

        keep = np.zeros_like(m, dtype=np.float32)
        if n_keep > 0:
            order = np.argsort(score)          # low->high
            keep_idx = active[order[-n_keep:]] # keep top
            keep[keep_idx] = 1.0

        mask_buffer[:] = keep.reshape(mask_buffer.shape)
        return mask_buffer

    def regrow_neurons(self, noRewires, shape, mask_buffer, extra_info=None):
        if noRewires <= 0:
            return

        inactive = np.argwhere(mask_buffer == 0)
        if inactive.shape[0] == 0:
            return

        if extra_info is None or ("g" not in extra_info) or ("v" not in extra_info):
            self._regrow_random(noRewires, shape, mask_buffer)
            return

        g = extra_info["g"]
        v = extra_info["v"]

        cand = min(inactive.shape[0], noRewires * 20)
        idx = self.rng.choice(inactive.shape[0], size=cand, replace=False)
        c = inactive[idx]
        ci, cj = c[:, 0], c[:, 1]

        score = (g[ci, cj] * g[ci, cj]) / (v[ci, cj] + self.eps)
        order = np.argsort(score)[::-1]

        grown = 0
        for k in order:
            i = int(ci[k]); j = int(cj[k])
            if mask_buffer[i, j] != 0:
                continue
            mask_buffer[i, j] = 1.0
            grown += 1
            if grown >= noRewires:
                return

        if grown < noRewires:
            self._regrow_random(noRewires - grown, shape, mask_buffer)

    def _regrow_random(self, k, shape, mask_buffer):
        rows, cols = shape
        tries = 0
        max_tries = k * 500
        grown = 0
        while grown < k and tries < max_tries:
            i = int(self.rng.integers(0, rows))
            j = int(self.rng.integers(0, cols))
            tries += 1
            if mask_buffer[i, j] != 0:
                continue
            mask_buffer[i, j] = 1.0
            grown += 1
