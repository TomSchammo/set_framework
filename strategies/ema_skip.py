import numpy as np
from .ema import NeuronEMASet


class NeuronEMASkipSET(NeuronEMASet):
    """
    EMA + SKIP (shared-budget chooser between W2 and Skip02)

    - W1 and W3 evolve exactly like NeuronEMASet via set_keras.rewireMask(...)
    - W2 and Skip02:
    """

    def __init__(self, *args, seed=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.rng = np.random.default_rng(seed)

    def choose_between_w2_and_skip(self, sf):
        """
        Called from set_keras.py in the "NeuronEMASkipSET" case.

        Requires sf to provide:
          - sf._prune_only_into_buffers(...)
          - sf.w2[0], sf.wSkip02[0]
          - sf.wm2_buffer/core, sf.wmSkip02_buffer/core
          - sf.noPar2, sf.noParSkip02
        """

        need2 = sf._prune_only_into_buffers(
            weights=sf.w2[0],
            noWeights=sf.noPar2,
            mask_buffer=sf.wm2_buffer,
            core_buffer=sf.wm2_core_buffer,
            extra_info={"layer": "layer_2", "self": sf},
            fisher_v=None,
        )

        needs = sf._prune_only_into_buffers(
            weights=sf.wSkip02[0],
            noWeights=sf.noParSkip02,
            mask_buffer=sf.wmSkip02_buffer,
            core_buffer=sf.wmSkip02_core_buffer,
            extra_info={"layer": "layer_2", "self": sf},
            fisher_v=None,
        )

        total_need = int(need2 + needs)
        if total_need <= 0:
            return

        # This computes EMA for 'layer_2' -> activation layer 'srelu2' -> length 1000
        self._update_ema_for_layer(sf, "layer_2")
        ema = self._ema.get("layer_2", None)

        inactive_w2 = np.argwhere(sf.wm2_buffer == 0)
        inactive_sk = np.argwhere(sf.wmSkip02_buffer == 0)

        if inactive_w2.shape[0] == 0 and inactive_sk.shape[0] == 0:
            return

        
        cand_total = min(total_need * 20, inactive_w2.shape[0] + inactive_sk.shape[0])
        cand_w2 = min(cand_total // 2, inactive_w2.shape[0])
        cand_sk = min(cand_total - cand_w2, inactive_sk.shape[0])

        merged = []


        def score_cols(cols):
            if ema is None or ema.size != 1000:
                return np.ones_like(cols, dtype=np.float64)
            return np.asarray(ema, dtype=np.float64)[cols]

   
        if cand_w2 > 0:
            idx = self.rng.choice(inactive_w2.shape[0], size=cand_w2, replace=False)
            c = inactive_w2[idx]
            ri, ci = c[:, 0], c[:, 1]
            s = score_cols(ci)
            for k in range(cand_w2):
                merged.append(("W2", int(ri[k]), int(ci[k]), float(s[k])))

        if cand_sk > 0:
            idx = self.rng.choice(inactive_sk.shape[0], size=cand_sk, replace=False)
            c = inactive_sk[idx]
            ri, ci = c[:, 0], c[:, 1]
            s = score_cols(ci)
            for k in range(cand_sk):
                merged.append(("SKIP", int(ri[k]), int(ci[k]), float(s[k])))

        for t in range(len(merged)):
            which, r, c, s = merged[t]
            merged[t] = (which, r, c, s + 1e-12 * self.rng.random())

        merged.sort(key=lambda t: t[3], reverse=True)
        merged = merged[:total_need]

        for which, r, c, _ in merged:
            if which == "W2":
                sf.wm2_buffer[r, c] = 1.0
            else:
                sf.wmSkip02_buffer[r, c] = 1.0
