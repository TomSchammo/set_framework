import numpy as np
import tensorflow as tf
from typing import Tuple, Optional, Set, List, Dict

from .base_strategy import BaseSETStrategy


def find_first_pos(array, value):
    idx = (np.abs(array - value)).argmin()
    return idx


def find_last_pos(array, value):
    idx = (np.abs(array - value))[::-1].argmin()
    return array.shape[0] - idx


class NeuronEMASet(BaseSETStrategy):
    """
    Pruning: same as RandomSET (legacy SET thresholding around 0).
    Regrow: biased toward neurons with high EMA(mean(|activation|)).
    """

    def __init__(self, zeta: float = 0.3, ema_beta: float = 0.9,
                 ema_batches: int = 1, sample_batch_size: int = 256,
                 eps: float = 1e-12):
        super().__init__(zeta)
        self.ema_beta = ema_beta
        self.ema_batches = ema_batches
        self.sample_batch_size = sample_batch_size
        self.eps = eps

        # layer_key -> ema vector (shape: [n_units])
        self._ema: Dict[str, np.ndarray] = {}
        # simple counter so we don't update EMA too often
        self._update_counter: Dict[str, int] = {}

    # -------------------------
    # Pruning = RandomSET prune
    # -------------------------
    def prune_neurons(self,
                      weight_values: np.ndarray,
                      weight_positions: Optional[np.ndarray] = None,
                      extra_info: Optional[dict] = None) -> np.ndarray:
        values = np.sort(weight_values)
        firstZeroPos = find_first_pos(values, 0)
        lastZeroPos = find_last_pos(values, 0)

        largestNegative = values[int((1 - self.zeta) * firstZeroPos)]
        smallestPositive = values[int(
            min(values.shape[0] - 1,
                lastZeroPos + self.zeta * (values.shape[0] - lastZeroPos)))
        ]

        return (weight_values > smallestPositive) | (weight_values < largestNegative)

    # -------------------------
    # EMA utilities
    # -------------------------
    def _get_activation_model(self, parent_model: tf.keras.Model, layer_name: str) -> tf.keras.Model:
        """
        Build a small model that outputs activations of a named layer.
        Cached by TF internally; cheap enough for our use.
        """
        lay = parent_model.get_layer(layer_name)
        return tf.keras.Model(inputs=parent_model.inputs, outputs=lay.output)

    def _update_ema_for_layer(self, parent, layer_key: str) -> None:
        """
        Updates EMA vector for a given layer_key using a small random batch.
        We read x_train from parent if available (minimal 1-line addition in set_keras).
        """
        # Expect parent to provide access to training data
        x_train = getattr(parent, "_ema_x_train", None)
        if x_train is None:
            # fallback: do nothing (strategy will degrade to uniform growth)
            return

        # map layer_key -> activation layer name in your Sequential model
        # layer_1 corresponds to srelu1 outputs, etc.
        layer_map = {
            "layer_1": "srelu1",
            "layer_2": "srelu2",
            "layer_3": "srelu3",
        }
        act_layer_name = layer_map.get(layer_key)
        if act_layer_name is None:
            return

        # throttle EMA updates
        c = self._update_counter.get(layer_key, 0) + 1
        self._update_counter[layer_key] = c
        if c % self.ema_batches != 0:
            return

        # sample batch
        bs = min(self.sample_batch_size, x_train.shape[0])
        idx = np.random.randint(0, x_train.shape[0], size=bs)
        xb = x_train[idx]

        act_model = self._get_activation_model(parent.model, act_layer_name)
        a = act_model(xb, training=False).numpy()  # shape: [bs, n_units]
        score = np.mean(np.abs(a), axis=0).astype(np.float64)  # [n_units]

        prev = self._ema.get(layer_key)
        if prev is None:
            self._ema[layer_key] = score
        else:
            self._ema[layer_key] = self.ema_beta * prev + (1.0 - self.ema_beta) * score

        # DEBUG: EMA update
        ema = self._ema[layer_key]
        print(
            f"[EMA][{layer_key}] update | "
            f"mean={ema.mean():.6e} "
            f"min={ema.min():.6e} "
            f"max={ema.max():.6e}"
        )

    def _sample_index_from_probs(self, probs: np.ndarray) -> int:
        probs = np.asarray(probs, dtype=np.float64)
        probs = probs + self.eps
        probs = probs / probs.sum()
        return int(np.random.choice(len(probs), p=probs))

    # -------------------------
    # Regrow = EMA-biased growth
    # -------------------------
    def regrow_neurons(self,
                       num_to_add: int,
                       dimensions: Tuple[int, int],
                       existing_positions: Set[Tuple[int, int]],
                       extra_info: Optional[dict] = None) -> List[Tuple[int, int]]:

        n_rows, n_cols = dimensions
        layer_key = None
        parent = None
        if extra_info:
            layer_key = extra_info.get("layer")
            parent = extra_info.get("self")

        # update EMA scores if we can
        if parent is not None and layer_key is not None:
            self._update_ema_for_layer(parent, layer_key)

        ema = self._ema.get(layer_key) if layer_key is not None else None

        positions: List[Tuple[int, int]] = []

        # If EMA not available, fall back to uniform sampling (still valid)
        if ema is None or len(ema) != n_cols:
            while len(positions) < num_to_add:
                i = np.random.randint(0, n_rows)
                j = np.random.randint(0, n_cols)
                if (i, j) not in existing_positions:
                    positions.append((i, j))
                    existing_positions.add((i, j))
            return positions

        # EMA-biased target neuron (column) sampling
        # Source neuron (row) can remain uniform (simple + minimal changes).
        while len(positions) < num_to_add:
            i = np.random.randint(0, n_rows)
            j = self._sample_index_from_probs(ema)  # bias toward active neurons
            if (i, j) not in existing_positions:
                positions.append((i, j))
                existing_positions.add((i, j))

        # ---- DEBUG: regrow targets bias ----
        if ema is not None and positions:
            cols = [j for (_, j) in positions]
            print(
                f"[EMA][{layer_key}] targets | "
                f"mean_target_ema={ema[cols].mean():.3e} "
                f"global_mean={ema.mean():.3e}"
            )
        # ----------------------------------

        return positions
