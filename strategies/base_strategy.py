from abc import ABC, abstractmethod
from typing import Optional, Tuple
import numpy as np


class BaseSETStrategy(ABC):
    """Abstract base for SET pruning/growth strategies"""

    def __init__(self, zeta: float = 0.3):
        self.zeta = zeta

    @abstractmethod
    def prune_neurons(self,
                      mask_buffer: np.ndarray,
                      weight_values: np.ndarray,
                      weight_positions: Optional[np.ndarray] = None,
                      extra_info: Optional[dict] = None) -> np.ndarray:
        """
        Decide which connections to KEEP.

        Args:
            mask_buffer: 2D array containing the current mask, to be modified in place
            weight_values: 1D array of weight values
            weight_positions: Optional (N, 2) array of (row, col) positions
            extra_info: Optional dict with gradients, activations, etc.

        Returns:
            Reference to the mask_buffer for API purposes

        """
        pass

    @abstractmethod
    def regrow_neurons(self,
                       num_to_add: int,
                       dimensions: Tuple[int, int],
                       mask: np.ndarray,
                       extra_info: Optional[dict] = None) -> None:
        """
        Decide WHERE to add new connections.

        Args:
            num_to_add: How many connections to add
            dimensions: (n_rows, n_cols) of weight matrix
            mask: 2D array containing current mask to be modified in place
            extra_info: Optional dict with neuron importance, etc.
        """
        pass
