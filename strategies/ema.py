import numpy as np
import tensorflow as tf
from typing import Tuple, Optional, Dict

from .base_strategy import BaseSETStrategy


def find_first_pos(array: np.ndarray, value: float) -> int:
    idx = (np.abs(array - value)).argmin()
    return int(idx)


def find_last_pos(array: np.ndarray, value: float) -> int:
    idx = (np.abs(array - value))[::-1].argmin()
    return int(array.shape[0] - idx)


class NeuronEMASet(BaseSETStrategy):
    """
    Pruning: RandomSET-style thresholding around 0 (applied only on existing edges).
    Regrow: biased toward neurons with high EMA(mean(|activation|)) of that layer.
    """

    def __init__(
        self,
        zeta: float = 0.3,
        ema_beta: float = 0.9,
        ema_batches: int = 1,
        sample_batch_size: int = 256,
        eps: float = 1e-12,
    ):
        super().__init__(zeta)

        #EMA must be updated every time
        assert ema_batches == 1, ("NeuronEMASet: ema_batches must be 1. "
                                  "EMA batching is disabled by design.")

        self.ema_beta = ema_beta
        self.ema_batches = ema_batches
        self.sample_batch_size = sample_batch_size
        self.eps = eps

        self._ema: Dict[str, np.ndarray] = {}
        self._update_counter: Dict[str, int] = {}
        self._act_model_cache: Dict[tuple[int, str], tf.keras.Model] = {}

    def prune_neurons(
        self,
        mask_buffer: np.ndarray,
        weight_values: np.ndarray,
        weight_positions: Optional[np.ndarray] = None,
        extra_info: Optional[dict] = None,
    ) -> np.ndarray:

        wflat = weight_values.ravel()

        mflat = mask_buffer.ravel()

        existing_idx = np.flatnonzero(mflat)
        if existing_idx.size == 0:
            raise RuntimeError(
                "NeuronEMASet.prune_neurons: layer has 0 existing connections. "
            )

        # Existing weights
        existing_w = wflat[existing_idx]

        values = np.sort(existing_w)

        first_zero = int(np.clip(find_first_pos(values, 0), 0,
                                 values.size - 1))
        last_zero = int(np.clip(find_last_pos(values, 0), 0, values.size))

        idx_neg = int(
            np.clip((1.0 - self.zeta) * first_zero, 0, values.size - 1))
        idx_pos = int(
            np.clip(last_zero + self.zeta * (values.size - last_zero), 0,
                    values.size - 1))

        largest_negative = values[idx_neg]
        smallest_positive = values[idx_pos]

        keep_existing = (existing_w > smallest_positive) | (existing_w
                                                            < largest_negative)

        # Drop pruned edges only
        prune_idx = existing_idx[~keep_existing]
        mask_buffer.flat[prune_idx] = 0

        return mask_buffer

    # utility functions
    def _get_activation_model(self, parent_model: tf.keras.Model,
                              layer_name: str) -> tf.keras.Model:
        key = (id(parent_model), layer_name)
        m = self._act_model_cache.get(key)
        if m is None:
            lay = parent_model.get_layer(layer_name)

            try:
                inputs = parent_model.input
            except AttributeError:
                inputs = parent_model.inputs

            m = tf.keras.Model(inputs=inputs, outputs=lay.output)
            self._act_model_cache[key] = m
        return m

    def _update_ema_for_layer(self, parent, layer_key: str) -> None:
        x_train = getattr(parent, "_ema_x_train", None)
        assert x_train is not None, (
            "NeuronEMASet: parent has no attribute '_ema_x_train'. "
            "EMA-based regrowth requires training data to be attached "
            "to the parent before calling regrow_neurons().")

        layer_map = {
            "layer_1": "srelu1",
            "layer_2": "srelu2",
            "layer_3": "srelu3"
        }
        act_layer_name = layer_map.get(layer_key)

        assert act_layer_name is not None, (
            f"NeuronEMASet: unknown layer_key '{layer_key}'. "
            f"Expected one of {list(layer_map.keys())}. "
            "EMA update cannot proceed without a valid activation layer mapping."
        )

        # Counter for logging, never skip updates
        c = self._update_counter.get(layer_key, 0) + 1
        self._update_counter[layer_key] = c

        bs = min(self.sample_batch_size, x_train.shape[0])
        idx = np.random.randint(0, x_train.shape[0], size=bs)
        xb = x_train[idx]

        act_model = self._get_activation_model(parent.model, act_layer_name)
        a = act_model([xb], training=False).numpy()
        score = np.mean(np.abs(a), axis=0).astype(np.float64)

        prev = self._ema.get(layer_key)
        if prev is None:
            self._ema[layer_key] = score
        else:
            self._ema[layer_key] = self.ema_beta * prev + (
                1.0 - self.ema_beta) * score

        ema = self._ema[layer_key]

        print(f"[EMA UPDATE] {layer_key} | "
              f"mean={ema.mean():.3e} "
              f"min={ema.min():.3e} "
              f"max={ema.max():.3e}")

    def _sample_col(self, probs: np.ndarray) -> int:
        p = np.asarray(probs, dtype=np.float64) + self.eps
        p /= p.sum()
        return int(np.random.choice(len(p), p=p))

    def regrow_neurons(
        self,
        num_to_add: int,
        dimensions: Tuple[int, int],
        mask: np.ndarray,
        extra_info: Optional[dict] = None,
    ) -> None:
        n_rows, n_cols = dimensions

        layer_key = extra_info.get("layer") if extra_info else None
        parent = extra_info.get("self") if extra_info else None

        if parent is not None and layer_key is not None:
            self._update_ema_for_layer(parent, layer_key)

        if layer_key is None:
            raise RuntimeError(
                "NeuronEMASet: missing layer key in extra_info['layer']; "
                "cannot do EMA-based regrowth without a layer id.")

        ema = self._ema.get(layer_key)

        if ema is None:
            raise RuntimeError(
                f"NeuronEMASet: EMA not initialized for layer '{layer_key}'. "
                "Run at least one EMA update before regrowth (ensure _ema_x_train is set and "
                "_update_ema_for_layer() is being called).")

        if ema.size != n_cols:
            raise RuntimeError(
                f"NeuronEMASet: EMA size mismatch for layer '{layer_key}': "
                f"ema.size={ema.size}, expected n_cols={n_cols}.")

        added = 0
        tries = 0
        max_tries = max(1000, num_to_add * 50)

        while added < num_to_add and tries < max_tries:
            tries += 1
            i = np.random.randint(0, n_rows)
            j = self._sample_col(ema)

            if mask[i, j] == 0:
                mask[i, j] = 1
                added += 1

        if added < num_to_add:
            raise RuntimeError(
                f"NeuronEMASet: could not regrow enough edges for layer '{layer_key}'. "
                f"requested={num_to_add}, added={added}, tries={tries}, max_tries={max_tries}."
            )
