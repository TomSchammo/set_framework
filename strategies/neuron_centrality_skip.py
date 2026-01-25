import numpy as np
from .neuron_centrality import NeuronCentralitySET


class NeuronCentralitySkipSET(NeuronCentralitySET):
    """
    Centrality + skip version.

    W1 and W3 are handled normally by set_keras (same as NeuronCentralitySET).
    For W2 and Skip02 we do a *shared regrowth budget*, analogous to FisherDiagonalSkipSET:
      - prune W2 and prune Skip02
      - compute total_need = need2 + needSkip
      - choose total_need new edges across BOTH matrices using centrality scores
    """

    def choose_between_w2_and_skip(self, sf, cand_mult=20):
        """
        sf is the SET_MLP_CIFAR10 instance (framework object).
        This method updates:
          - sf.wm2_buffer, sf.wm2_core_buffer
          - sf.wmSkip02_buffer, sf.wmSkip02_core_buffer
        in-place (buffer mindset).
        """
        # --- prune W2 into its buffer, snapshot core ---
        # prune_neurons(mask_buffer, weight_values, weight_positions, extra_info)
        self.prune_neurons(sf.wm2_buffer, sf.w2[0].ravel(), sf.wm2_buffer.ravel(), extra_info=None)
        np.copyto(sf.wm2_core_buffer, sf.wm2_buffer)
        need2 = int(sf.noPar2 - np.sum(sf.wm2_buffer))

        # --- prune skip into its buffer, snapshot core ---
        self.prune_neurons(sf.wmSkip02_buffer, sf.wSkip02[0].ravel(), sf.wmSkip02_buffer.ravel(), extra_info=None)
        np.copyto(sf.wmSkip02_core_buffer, sf.wmSkip02_buffer)
        needS = int(sf.noParSkip02 - np.sum(sf.wmSkip02_buffer))

        total_need = need2 + needS
        if total_need <= 0:
            return

        # inactive positions after pruning
        inactive2 = np.argwhere(sf.wm2_buffer == 0)
        inactiveS = np.argwhere(sf.wmSkip02_buffer == 0)
        if inactive2.shape[0] == 0 and inactiveS.shape[0] == 0:
            return

        # --- compute centrality importance vectors ---
        # target for both W2 and Skip is layer2 neurons (size 1000)
        I_target = self.hidden_layer_neuron_importance(
            W_in=sf.w2[0], M_in=sf.wm2_buffer,
            W_out=sf.w3[0], M_out=sf.wm3_buffer
        )  # (1000,)

        # source for W2 is layer1 neurons (size 4000)
        I_source_w2 = self.hidden_layer_neuron_importance(
            W_in=sf.w1[0], M_in=sf.wm1_buffer,
            W_out=sf.w2[0], M_out=sf.wm2_buffer
        )  # (4000,)

        # source for skip is input neurons (size 3072)
        I_source_skip = self.input_layer_importance(sf.w1[0], mask=sf.wm1_buffer)  # (3072,)

        alpha = float(getattr(self, "alpha", 0.7))
        eps = float(getattr(self, "eps", 1e-12))

        # candidate sampling
        cand_total = min(total_need * cand_mult, inactive2.shape[0] + inactiveS.shape[0])
        cand2 = min(cand_total // 2, inactive2.shape[0])
        candS = min(cand_total - cand2, inactiveS.shape[0])

        merged = []

        rng = np.random.default_rng()

        if cand2 > 0:
            idx = rng.choice(inactive2.shape[0], size=cand2, replace=False)
            c = inactive2[idx]
            ri, cj = c[:, 0], c[:, 1]
            # score = alpha * I_target[col] * I_source[row] + (1-alpha)*1
            score = alpha * (I_target[cj] * I_source_w2[ri]) + (1.0 - alpha) * 1.0
            for k in range(cand2):
                merged.append(("W2", int(ri[k]), int(cj[k]), float(score[k])))

        if candS > 0:
            idx = rng.choice(inactiveS.shape[0], size=candS, replace=False)
            c = inactiveS[idx]
            ri, cj = c[:, 0], c[:, 1]
            score = alpha * (I_target[cj] * I_source_skip[ri]) + (1.0 - alpha) * 1.0
            for k in range(candS):
                merged.append(("WS", int(ri[k]), int(cj[k]), float(score[k])))

        if len(merged) == 0:
            return

        scores = np.array([m[3] for m in merged], dtype=np.float64)
        scores = np.clip(scores, 0.0, None)
        ssum = scores.sum()

        if not np.isfinite(ssum) or ssum <= eps:
            probs = None  # uniform
        else:
            probs = scores / ssum

        choose_k = min(total_need, len(merged))
        chosen_idx = rng.choice(len(merged), size=choose_k, replace=False, p=probs)

        for idx in chosen_idx:
            which, i, j, _ = merged[idx]
            if which == "W2":
                sf.wm2_buffer[i, j] = 1.0
            else:
                sf.wmSkip02_buffer[i, j] = 1.0
