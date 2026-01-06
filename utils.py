"""
Temporary file to extract things that both keras and sparse implementations have in common
"""

import numpy as np
from typing import Tuple


def find_first_pos(array, value):
    idx = (np.abs(array - value)).argmin()
    return idx


def find_last_pos(array, value):
    idx = (np.abs(array - value))[::-1].argmin()
    return array.shape[0] - idx


def compute_removal_thresholds(weight_values: np.ndarray,
                               zeta: float) -> Tuple[float, float]:
    """Pure function: given weights, compute what to keep"""
    values = np.sort(weight_values)
    firstZeroPos = find_first_pos(values, 0)
    lastZeroPos = find_last_pos(values, 0)
    largestNegative = values[int((1 - zeta) * firstZeroPos)]
    smallestPositive = values[int(
        min(values.shape[0] - 1,
            lastZeroPos + zeta * (values.shape[0] - lastZeroPos)))]
    return largestNegative, smallestPositive


# TODO not sure about this one
def compute_keep_mask(weight_values: np.ndarray, lower_threshold: float,
                      upper_threshold: float) -> np.ndarray:
    """Pure function: returns boolean mask of what to keep"""
    return (weight_values > upper_threshold) | (weight_values
                                                < lower_threshold)
