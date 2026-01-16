from abc import ABC, abstractmethod
from typing import Optional, Set, Tuple, List
import numpy as np
import torch


class BaseSETStrategy(ABC):
    """Abstract base for SET pruning/growth strategies"""

    def __init__(self, zeta: float = 0.3):
        self.zeta = zeta

    @abstractmethod
    def prune_neurons(self,
                      weight_values: np.ndarray,
                      weight_positions: Optional[np.ndarray] = None,
                      extra_info: Optional[dict] = None) -> np.ndarray:
        """
        Decide which connections to KEEP.

        Args:
            weight_values: 1D array of weight values
            weight_positions: Optional (N, 2) array of (row, col) positions
            extra_info: Optional dict with gradients, activations, etc.

        Returns:
            Boolean array: True = keep, False = remove
        """
        pass

    @abstractmethod
    def prune_neurons_gpu(self,
                          weight_values: torch.Tensor,
                          weight_positions: Optional[torch.Tensor] = None,
                          extra_info: Optional[dict] = None) -> torch.Tensor:
        """
        Decide which connections to KEEP.

        Args:
            weight_values: 1D array of weight values
            weight_positions: Optional (N, 2) array of (row, col) positions
            extra_info: Optional dict with gradients, activations, etc.

        Returns:
            Boolean array: True = keep, False = remove
        """
        pass

    @abstractmethod
    def regrow_neurons(
            self,
            num_to_add: int,
            dimensions: Tuple[int, int],
            existing_positions: Set[Tuple[int, int]],
            extra_info: Optional[dict] = None) -> List[Tuple[int, int]]:
        """
        Decide WHERE to add new connections.

        Args:
            num_to_add: How many connections to add
            dimensions: (n_rows, n_cols) of weight matrix
            existing_positions: Set of (row, col) that already exist
            extra_info: Optional dict with neuron importance, etc.

        Returns:
            List of (row, col) tuples for new connections
        """
        pass

    @abstractmethod
    def regrow_neurons_gpu(self,
                           num_to_add: int,
                           dimensions: torch.Size,
                           existing_positions: torch.Tensor,
                           extra_info: Optional[dict] = None) -> torch.Tensor:
        """
        Decide WHERE to add new connections.

        Args:
            num_to_add: How many connections to add
            dimensions: (n_rows, n_cols) of weight matrix
            existing_positions: Set of (row, col) that already exist
            extra_info: Optional dict with neuron importance, etc.

        Returns:
            List of (row, col) tuples for new connections
        """
        pass
