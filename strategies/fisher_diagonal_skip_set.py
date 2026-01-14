import numpy as np
from strategies.base_strategy import BaseSETStrategy


class FisherDiagonalSkipSET(BaseSETStrategy):
    """
    Fisher diagonal + skip-layer strategy.

    Assumptions:
      - set_keras computes gradients g and maintains V (EMA of g^2),
        then passes V into prune (extra_info['v_flat']) and passes g+V
        into regrow (extra_info['g'], extra_info['v']).

    Scores:
      - prune score (keep-utility / "harm if removed"):
            S_prune = w^2 / (V + eps)
        prune removes lowest zeta fraction, keeps highest.

      - regrow score for missing edges:
            S_grow = g^2 / (V + eps)
        regrow picks highest-scoring candidate positions.
    """

    needs_gradients = True  # tells set_keras to compute grads and update V

    def __init__(self, zeta=0.3, beta=0.9, eps=1e-8, seed=None):
        self.zeta = float(zeta)
        self.beta = float(beta)
        self.eps = float(eps)
        self.rng = np.random.default_rng(seed)

        # Fisher diagonal (EMA of grad^2) per matrix (set_keras fills these)
        self.v1 = None
        self.v2 = None
        self.v3 = None
        self.vSkip02 = None

    def prune_neurons(self, weights_flat: np.ndarray, mask_flat: np.ndarray, extra_info=None):
        """
        Return a NEW mask (flat) with same shape, 1 = keep, 0 = prune.
        Only considers currently ACTIVE edges (mask==1) for pruning.
        """
        if extra_info is None or "v_flat" not in extra_info:
            raise ValueError("FisherDiagonalSkipSET.prune_neurons requires extra_info['v_flat'].")

        w = weights_flat.astype(np.float32, copy=False)
        m = mask_flat.astype(np.float32, copy=False)
        v = extra_info["v_flat"].astype(np.float32, copy=False)

        active = np.flatnonzero(m > 0.0)
        if active.size == 0:
            return m.copy()

        n_active = active.size
        n_keep = int(np.round((1.0 - self.zeta) * n_active))
        n_keep = max(0, min(n_active, n_keep))

        # keep-utility score (as requested)
        score = (w[active] * w[active]) / (v[active] + self.eps)

        keep = np.zeros_like(m, dtype=np.float32)
        if n_keep > 0:
            # keep highest scores, prune lowest
            order = np.argsort(score)             # low -> high
            keep_idx = active[order[-n_keep:]]    # take top n_keep
            keep[keep_idx] = 1.0

        return keep

    def regrow_neurons(self, noRewires: int, shape, occupied, extra_info=None):
        """
        Pick new (i,j) positions to activate (regrow).
        Uses score S_grow = g^2/(v+eps) on candidates.
        """
        if noRewires <= 0:
            return []

        if extra_info is None or ("g" not in extra_info) or ("v" not in extra_info) or ("mask" not in extra_info):
            # fallback random
            return self._regrow_random(noRewires, shape[0], shape[1], occupied)

        g = extra_info["g"]
        v = extra_info["v"]
        mask_bool = extra_info["mask"]  # True where active

        rows, cols = shape

        inactive = np.argwhere(~mask_bool)
        if inactive.shape[0] == 0:
            return []

        # sample candidates (bounded)
        cand = min(inactive.shape[0], noRewires * 20)
        idx = self.rng.choice(inactive.shape[0], size=cand, replace=False)
        c = inactive[idx]
        ci, cj = c[:, 0], c[:, 1]

        score = (g[ci, cj] * g[ci, cj]) / (v[ci, cj] + self.eps)

        # pick best candidates
        order = np.argsort(score)[::-1]  # high -> low
        new_pos = []
        for k in order:
            i = int(ci[k])
            j = int(cj[k])
            if (i, j) in occupied:
                continue
            occupied.add((i, j))
            new_pos.append((i, j))
            if len(new_pos) >= noRewires:
                break

        # fallback if not enough
        if len(new_pos) < noRewires:
            new_pos.extend(self._regrow_random(noRewires - len(new_pos), rows, cols, occupied))

        return new_pos

    def _regrow_random(self, k, rows, cols, occupied):
        out = []
        tries = 0
        max_tries = k * 200
        while len(out) < k and tries < max_tries:
            i = int(self.rng.integers(0, rows))
            j = int(self.rng.integers(0, cols))
            tries += 1
            if (i, j) in occupied:
                continue
            occupied.add((i, j))
            out.append((i, j))
        return out
