import numpy as np
import tensorflow as tf
from typing import Tuple, Optional, Dict, Any

from .base_strategy import BaseSETStrategy


class NeuronEMASET(BaseSETStrategy):
    """
    Buffer / in-place EMA-biased SET.

    Prune: smallest-magnitude fraction among ACTIVE edges (in-place).
    Regrow: choose new edges biased toward high EMA(mean(|activation|)) of TARGET neurons.
            Supports layers: layer_1, layer_2, layer_3, skip_02

    Notes:
      - Expects set_keras to set parent._ema_x_train = x_train
    """

    def __init__(
        self,
        zeta: float = 0.3,
        ema_beta: float = 0.9,
        ema_batches: int = 1,
        sample_batch_size: int = 256,
        eps: float = 1e-12,
        seed=None,
    ):
        super().__init__(float(zeta))
        self.ema_beta = float(ema_beta)
        self.ema_batches = int(ema_batches)
        self.sample_batch_size = int(sample_batch_size)
        self.eps = float(eps)
        self.rng = np.random.default_rng(seed)

        self._ema: Dict[str, np.ndarray] = {}
        self._counter: Dict[str, int] = {}

        
    def prune_neurons(
        self,
        mask_buffer: np.ndarray,
        weight_values: np.ndarray,
        weight_positions: Optional[np.ndarray] = None,
        extra_info: Optional[Dict[str, Any]] = None,
    ) -> np.ndarray:
        mb = mask_buffer.ravel()
        w = np.asarray(weight_values).ravel()
        if weight_positions is None:
            m = (mb != 0)
        else:
            m = np.asarray(weight_positions).ravel().astype(bool)

        if mb.shape != w.shape or mb.shape != m.shape:
            raise ValueError(f"Shape mismatch: mb{mb.shape}, w{w.shape}, m{m.shape}")

        np.copyto(mb, m.astype(mb.dtype, copy=False))

        active = np.flatnonzero(m)
        n_active = active.size
        if n_active == 0:
            mb[:] = 0
            return mask_buffer

        k = int(self.zeta * n_active)
        if k <= 0:
            return mask_buffer

        vals = np.abs(w[active])
        order = np.argsort(vals)  # low -> high
        prune_idx = active[order[:k]]
        mb[prune_idx] = 0
        return mask_buffer

    
    def _activation_model(self, parent_model: tf.keras.Model, layer_name: str) -> tf.keras.Model:
        lay = parent_model.get_layer(layer_name)
        return tf.keras.Model(inputs=parent_model.inputs, outputs=lay.output)

    def _update_ema(self, parent, layer_key: str) -> None:
        x_train = getattr(parent, "_ema_x_train", None)
        if x_train is None:
            return

        # map edge-layer -> activation layer to measure TARGET neuron activity
        layer_map = {
            "layer_1": "srelu1",  # target = 4000
            "layer_2": "srelu2",  # target = 1000
            "layer_3": "srelu3",  # target = 4000
            "skip_02": "srelu2",  # skip targets layer2 units (1000)
        }
        act_layer_name = layer_map.get(layer_key)
        if act_layer_name is None:
            return

        c = self._counter.get(layer_key, 0) + 1
        self._counter[layer_key] = c
        if c % self.ema_batches != 0:
            return

        bs = min(self.sample_batch_size, x_train.shape[0])
        idx = self.rng.integers(0, x_train.shape[0], size=bs)
        xb = x_train[idx]

        act_model = self._activation_model(parent.model, act_layer_name)
        a = act_model([xb], training=False).numpy()  
        score = np.mean(np.abs(a), axis=0).astype(np.float64)

        prev = self._ema.get(layer_key)
        if prev is None:
            self._ema[layer_key] = score
        else:
            self._ema[layer_key] = self.ema_beta * prev + (1.0 - self.ema_beta) * score

            
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

        parent = None
        layer_key = None
        if extra_info:
            parent = extra_info.get("self")
            layer_key = extra_info.get("layer")

        if parent is None or layer_key is None:
            self._regrow_uniform(k, dimensions, mask)
            return

        self._update_ema(parent, layer_key)
        ema = self._ema.get(layer_key)

        # if EMA missing or wrong size -> uniform
        if ema is None or ema.size != n_cols:
            self._regrow_uniform(k, dimensions, mask)
            return

        p = np.clip(ema.astype(np.float64), 0.0, None) + self.eps
        p = p / p.sum()

        added = 0
        attempts = 0
        max_attempts = max(10_000, 50 * k)

        while added < k and attempts < max_attempts:
            attempts += 1
            i = int(self.rng.integers(0, n_rows))        # source uniform
            j = int(self.rng.choice(n_cols, p=p))        # target EMA-biased
            if mask[i, j] == 0:
                mask[i, j] = 1
                added += 1

        if added < k:
            self._regrow_uniform(k - added, dimensions, mask)

    def _regrow_uniform(self, k: int, dimensions: Tuple[int, int], mask: np.ndarray) -> None:
        n_rows, n_cols = dimensions
        added = 0
        attempts = 0
        max_attempts = max(10_000, 50 * k)
        while added < k and attempts < max_attempts:
            attempts += 1
            i = int(self.rng.integers(0, n_rows))
            j = int(self.rng.integers(0, n_cols))
            if mask[i, j] == 0:
                mask[i, j] = 1
                added += 1
