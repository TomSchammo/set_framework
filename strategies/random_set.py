import numpy as np
from typing import Tuple, Optional, Set, List
import torch

from .base_strategy import BaseSETStrategy


def find_first_pos(array, value):
    idx = (np.abs(array - value)).argmin()
    return idx


def find_last_pos(array, value):
    idx = (np.abs(array - value))[::-1].argmin()
    return array.shape[0] - idx


def find_first_pos_gpu(array: torch.Tensor, value):
    idx = (torch.abs(array - value)).argmin()
    return idx.item()


def find_last_pos_gpu(array: torch.Tensor, value):
    idx = torch.abs(torch.flip(array, dims=[0]) - value).argmin()
    return (array.shape[0] - idx).item()


class RandomSET(BaseSETStrategy):

    def __init__(self, zeta: float = 0.3):
        super().__init__(zeta)

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
                lastZeroPos + self.zeta * (values.shape[0] - lastZeroPos)))]

        return (weight_values > smallestPositive) | (weight_values
                                                     < largestNegative)

    def prune_neurons_gpu(self,
                          weight_values: torch.Tensor,
                          weight_positions: Optional[torch.Tensor] = None,
                          extra_info: Optional[dict] = None) -> torch.Tensor:
        values, _ = torch.sort(weight_values)
        firstZeroPos = find_first_pos_gpu(values, 0)
        lastZeroPos = find_last_pos_gpu(values, 0)

        values = values.reshape(-1)
        largestNegative = values[int((1 - self.zeta) * firstZeroPos)]
        smallestPositive = values[int(
            min(values.shape[0] - 1,
                lastZeroPos + self.zeta * (values.shape[0] - lastZeroPos)))]

        return (weight_values > smallestPositive) | (weight_values
                                                     < largestNegative)

    def regrow_neurons_gpu(self,
                           num_to_add: int,
                           dimensions: torch.Size,
                           existing_positions: torch.Tensor,
                           extra_info: Optional[dict] = None) -> torch.Tensor:

        device = existing_positions.device
        dtype = existing_positions.dtype
        positions = torch.empty((0, 2), device=device, dtype=dtype)

        while positions.size(0) < num_to_add:
            i = torch.randint(0,
                              dimensions[0], (1, ),
                              device=device,
                              dtype=dtype)
            j = torch.randint(0,
                              dimensions[1], (1, ),
                              device=device,
                              dtype=dtype)
            candidate = torch.stack((i, j), dim=1)  # shape: (1, 2)

            if existing_positions.numel() == 0:
                is_new = True
            else:
                is_new = ~((existing_positions == candidate).all(dim=1)).any()

            if is_new:
                positions = torch.cat((positions, candidate), dim=0)
                existing_positions = (candidate if existing_positions.numel()
                                      == 0 else torch.cat(
                                          (existing_positions, candidate),
                                          dim=0))

        return positions

    def regrow_neurons(
            self,
            num_to_add: int,
            dimensions: Tuple[int, int],
            existing_positions: Set[Tuple[int, int]],
            extra_info: Optional[dict] = None) -> List[Tuple[int, int]]:
        positions = []
        while len(positions) < num_to_add:
            i = np.random.randint(0, dimensions[0])
            j = np.random.randint(0, dimensions[1])
            if (i, j) not in existing_positions:
                positions.append((i, j))
                existing_positions.add((i, j))
        return positions
