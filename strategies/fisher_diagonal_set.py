import numpy as np
from typing import Tuple, Optional, Dict, Any
from .base_strategy import BaseSETStrategy


class FisherDiagonalSET(BaseSETStrategy):
    """
    Buffer/in-place Fisher diagonal SET (with optional skip via uses_skip flag).

    prune score (keep utility): w^2 / (V + eps)
    regrow score:              g^2 / (V + eps)

    set_keras passes:
      prune: extra_info["v_flat"]  (len = weights.ravel())
      grow : extra_info["g"], extra_info["v"] (2D arrays same shape as mask)
    """

    needs_gradients = True

    def __init__(self, zeta=0.3, beta=0.9, eps=1e-8, seed=None, use_skip: bool = False):
        super().__init__(zeta=float(zeta))
        self.beta = float(beta)
        self.eps = float(eps)
        self.rng = np.random.default_rng(seed)
        self.uses_skip = bool(use_skip)

        # set_keras will create/update these
        self.v1 = None
        self.v2 = None
        self.v3 = None
        self.vSkip02 = None

    def prune_neurons(
        self,
        mask_buffer: np.ndarray,
        weight_values: np.ndarray,
        weight_positions: Optional[np.ndarray] = None,
        extra_info: Optional[Dict[str, Any]] = None,
    ) -> np.ndarray:
        if extra_info is None or "v_flat" not in extra_info:
            raise ValueError("FisherDiagonalSET.prune_neurons requires extra_info['v_flat'].")

        mb = mask_buffer.ravel()
        w = np.asarray(weight_values).ravel().astype(np.float32, copy=False)

        if weight_positions is None:
            m = (w != 0)
        else:
            m = np.asarray(weight_positions).ravel().astype(bool)

        v = np.asarray(extra_info["v_flat"]).ravel().astype(np.float32, copy=False)

        if mb.shape != w.shape or m.shape != w.shape or v.shape != w.shape:
            raise ValueError(f"Shape mismatch: mb{mb.shape} w{w.shape} m{m.shape} v{v.shape}")

        np.copyto(mb, m.astype(mb.dtype, copy=False))

        active = np.flatnonzero(m)
        n_active = active.size
        if n_active == 0:
            mb[:] = 0
            return mask_buffer

        n_keep = int(np.round((1.0 - self.zeta) * n_active))
        n_keep = max(0, min(n_active, n_keep))

        mb[:] = 0
        if n_keep == 0:
            return mask_buffer

        score = (w[active] * w[active]) / (v[active] + self.eps)
        order = np.argsort(score)            # low -> high
        keep_idx = active[order[-n_keep:]]   # keep top n_keep
        mb[keep_idx] = 1
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

        n_rows, n_cols = dimensions

        if extra_info is None or ("g" not in extra_info) or ("v" not in extra_info):
            self._regrow_random_inplace(k, mask)
            return

        g = np.asarray(extra_info["g"], dtype=np.float32)
        v = np.asarray(extra_info["v"], dtype=np.float32)

        if g.shape != (n_rows, n_cols) or v.shape != (n_rows, n_cols):
            self._regrow_random_inplace(k, mask)
            return

        inactive = np.argwhere(mask == 0)
        N = inactive.shape[0]
        if N == 0:
            return
        k = min(k, N)

        # score candidates; bound candidate count for speed
        cand = min(N, k * 20)
        idx = self.rng.choice(N, size=cand, replace=False)
        c = inactive[idx]
        ci, cj = c[:, 0], c[:, 1]

        score = (g[ci, cj] * g[ci, cj]) / (v[ci, cj] + self.eps)
        order = np.argsort(score)[::-1]  # high -> low
        chosen = c[order[:k]]

        mask[chosen[:, 0], chosen[:, 1]] = 1

    def _regrow_random_inplace(self, k: int, mask: np.ndarray) -> None:
        inactive = np.argwhere(mask == 0)
        if inactive.shape[0] == 0:
            return
        k = min(k, inactive.shape[0])
        idx = self.rng.choice(inactive.shape[0], size=k, replace=False)
        pick = inactive[idx]
        mask[pick[:, 0], pick[:, 1]] = 1
