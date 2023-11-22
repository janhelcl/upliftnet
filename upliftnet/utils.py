"""
# TODO: docs
"""
import numpy as np


def check_is_binary(to_check: np.ndarray):
    """
    Checks if the array contains only ones and zeros (both ints and floats are acceptable)
    
    :raises: ValueError if conditions are not met
    """
    if not np.all(np.unique(to_check) == np.array([0, 1])):
        raise ValueError(f"Input array is not binary. "
                         f"Array should contain only int or float binary values 0 (or 0.) and 1 (or 1.). "
                         f"Got values {np.unique(to_check)}.")


def check_is_probability(to_check: np.ndarray):
    """
    Check if 0.0 <= value <= 1.0 for all values in input array
    
    :raises: ValueError if conditions are not met
    """
    min_val = to_check.min()
    max_val = to_check.max()
    if min_val < 0 or 1 < max_val:
        raise ValueError("Input array does not contain valid probabilities. "
                         f"Maximum value found: {max_val}, minimum value found: {min_val}.")
