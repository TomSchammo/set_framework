import numpy as np
from typing import Tuple, Optional, Set, List

from base_strategy import BaseSETStrategy


def find_first_pos(array, value):
    idx = (np.abs(array - value)).argmin()
    return idx


def find_last_pos(array, value):
    idx = (np.abs(array - value))[::-1].argmin()
    return array.shape[0] - idx


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
