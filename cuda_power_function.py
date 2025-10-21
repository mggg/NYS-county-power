"""
This is a file for containing the cuda version of the power function for use with 
NVIDIA GPUs. This can provide a 3x speedup over the numpy version.
"""

import cupy as cp


def compute_power_cupy(
    u_matrix: cp.ndarray,
    subset_masks_bool: cp.ndarray,
    subset_masks_float: cp.ndarray,
    T: float,
):
    """
    This function computes the difference between the proportion of voting power for a locality
    and the proportion of the population that town has.

    Equivalent to `compute_power_cupy_simple` with the exception that the subsets in
    `subset_masks_bool` and `subset_masks_float` assume that one town, in particular, always votes
    in the affirmative. This reduces the memory cost of the computation but otherwise does not
    affect the speed significantly.

    Args:
        u_matrix (cupy.ndarray): Array of size (num_towns, 1) that contains the weights to assign
            to each town.
        subset_masks_bool (cupy.ndarray): Array of size (num_towns, num_subsets) that contains
            boolean masks for whether a town voted in favor of a particular measure. It is assumed
            that one town, in particular, always votes in the affirmative.
        subset_masks_float (cupy.ndarray): Array of size (num_towns, num_subsets) that contains
            the same data as the boolean masks, but as floats for ease of computation.
        T (float): The threshold for the power.

    Returns:
        cupy.ndarray:
            An array of size (num_towns,) that contains the computed voting power for when each
            town is given weight determined by the passed u_matrix.
    """
    totu = u_matrix.sum()
    T = T * totu
    a_matrix = cp.transpose(u_matrix) @ subset_masks_float

    a_minus_T = a_matrix - T
    totu_minus_a = totu - a_matrix
    totu_minus_a_minus_T = totu_minus_a - T

    condition1 = cp.logical_and(u_matrix >= a_minus_T, a_matrix > T)
    condition2 = cp.logical_and(u_matrix >= totu_minus_a_minus_T, totu_minus_a > T)

    X = cp.logical_or(
        cp.logical_and(subset_masks_bool, condition1),
        cp.logical_and(~subset_masks_bool, condition2),
    )

    non_zero_count = cp.count_nonzero(X, axis=1)
    p = non_zero_count / cp.count_nonzero(X)

    return p


def compute_power_cupy_simple(
    u_matrix: cp.ndarray,
    subset_masks_bool: cp.ndarray,
    subset_masks_float: cp.ndarray,
    T: float,
):
    """
    This function computes the difference between the proportion of voting power for a locality
    and the proportion of the population that town has.

    Equivalent to `compute_power_cupy` with the exception that the subsets in `subset_masks_bool`
    and `subset_masks_float` include all possible, non-empty, affirmative coalitions in this
    implementation.

    Note: This is slower than the `compute_power_cupy` function.

    Args:
        u_matrix (cupy.ndarray): Array of size (num_towns, 1) that contains the weights to assign
            to each town.
        subset_masks_bool (cupy.ndarray): Array of size (num_towns, num_subsets) that contains
            boolean masks for whether a town voted in favor of a particular measure. It is assumed
            that all subsets with at least one town voting in the affirmative are included.
        subset_masks_float (cupy.ndarray): Array of size (num_towns, num_subsets) that contains
            the same data as the boolean masks, but as floats for ease of computation.
        T (float): The threshold for the power.

    Returns:
        cupy.ndarray:
            An array of size (num_towns,) that contains the computed voting power for when each
            town is given weight determined by the passed u_matrix.
    """
    totu = u_matrix.sum()
    T = T * totu
    a_matrix = cp.transpose(u_matrix) @ subset_masks_float

    a_minus_T = a_matrix - T

    X = cp.logical_and(
        subset_masks_bool, cp.logical_and(u_matrix >= a_minus_T, a_matrix > T)
    )

    non_zero_count = cp.count_nonzero(X, axis=1)
    p = non_zero_count / cp.count_nonzero(X)

    return p


def compute_power_cupy_simple_dyn_tau(
    u_matrix: cp.ndarray,
    subset_masks_bool: cp.ndarray,
    subset_masks_float: cp.ndarray,
    T: float,
    numerator: int,
):
    """
    This function computes the difference between the proportion of voting power for a locality
    and the proportion of the population that town has.

    Equivalent to `compute_power_cupy` with the exception that the subsets in `subset_masks_bool`
    and `subset_masks_float` include all possible, non-empty, affirmative coalitions in this
    implementation.

    Note: This is slower than the `compute_power_cupy` function.

    Args:
        u_matrix (cupy.ndarray): Array of size (num_towns, 1) that contains the weights to assign
            to each town.
        subset_masks_bool (cupy.ndarray): Array of size (num_towns, num_subsets) that contains
            boolean masks for whether a town voted in favor of a particular measure. It is assumed
            that all subsets with at least one town voting in the affirmative are included.
        subset_masks_float (cupy.ndarray): Array of size (num_towns, num_subsets) that contains
            the same data as the boolean masks, but as floats for ease of computation.
        T (float): The threshold for the power.

    Returns:
        cupy.ndarray:
            An array of size (num_towns,) that contains the computed voting power for when each
            town is given weight determined by the passed u_matrix.
    """
    totu = u_matrix.sum()
    T = T * totu
    a_matrix = cp.transpose(u_matrix) @ subset_masks_float

    a_minus_T = a_matrix - T

    X = cp.logical_and(
        subset_masks_bool, cp.logical_and(u_matrix >= a_minus_T, a_matrix > T)
    )

    base = cp.asarray(numerator, dtype=cp.float64)
    yes_weights = cp.power(base, subset_masks_bool.sum(axis=0))
    weighted_counts_matrix = cp.multiply(X, yes_weights).T

    non_zero_count = weighted_counts_matrix.sum(axis=0)
    p = non_zero_count / cp.sum(non_zero_count)

    return p
