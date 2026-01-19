import numpy as np
from typing import Tuple, Optional, Set, List

from .base_strategy import BaseSETStrategy


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
                      mask_buffer: np.ndarray,
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

        pid = id(mask_buffer)
        pid_addr = mask_buffer.__array_interface__['data'][0]

        mask_buffer_flat = mask_buffer.ravel()

        assert pid_addr == mask_buffer_flat.__array_interface__['data'][
            0], "flatten resulted copy"

        assert extra_info

        temp_buffer = extra_info['temp_buf'].ravel()

        np.greater(weight_values, smallestPositive, out=mask_buffer_flat)
        np.less(weight_values, largestNegative, out=temp_buffer)
        np.logical_or(mask_buffer_flat, temp_buffer, out=mask_buffer_flat)

        assert pid == id(
            mask_buffer), "mask buffer has been rebound during prune"

        return mask_buffer

    def regrow_neurons(self,
                       num_to_add: int,
                       dimensions: Tuple[int, int],
                       mask: np.ndarray,
                       extra_info: Optional[dict] = None) -> None:

        count = 0
        while count < num_to_add:
            i = np.random.randint(0, dimensions[0])
            j = np.random.randint(0, dimensions[1])

            if mask[i, j] == 0:
                mask[i, j] = 1
                count += 1
