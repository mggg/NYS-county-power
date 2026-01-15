import numpy as np


def compute_power_numpy(
    u_matrix: np.ndarray,
    subset_masks: np.ndarray,
    T: float,
):
    """
    This function computes the difference between the proportion of voting power for a locality
    and the proportion of the population that town has.

    Equivalent to `compute_power_numpy_simple` with the exception that the subsets in `subset_masks`
    assumes that one town, in particular, always votes in the affirmative. This reduces the
    memory cost of the computation but otherwise does not affect the speed significantly.

    Args:
        u_matrix (numpy.ndarray): Array of size (num_towns, 1) that contains the weights to assign
            to each town.
        subset_masks (numpy.ndarray): Array of size (num_towns, num_subsets) that contains
            boolean masks for whether a town voted in favor of a particular measure. Note:
            it is generally assumed that one town, in particular, always votes in the affirmative
            in this version.
        T (float): The threshold for the power.

    Returns:
        numpy.ndarray:
            An array of size (num_towns,) that contains the computed voting power for when each
            town is given weight determined by the u_matrix.
    """
    totu = u_matrix.sum()
    T = T * totu
    a_matrix = np.transpose(u_matrix) @ subset_masks

    a_minus_T = a_matrix - T
    totu_minus_a = totu - a_matrix
    totu_minus_a_minus_T = totu_minus_a - T

    condition1 = np.logical_and(u_matrix >= a_minus_T, a_matrix > T)
    condition2 = np.logical_and(u_matrix >= totu_minus_a_minus_T, totu_minus_a > T)

    X = np.logical_or(
        np.logical_and(subset_masks, condition1),
        np.logical_and(~subset_masks, condition2),
    )

    non_zero_count = np.count_nonzero(X, axis=1)
    p = non_zero_count / np.count_nonzero(X)

    return p


def compute_power_numpy_simple(
    u_matrix: np.ndarray,
    subset_masks: np.ndarray,
    T: float,
):
    """
    This function computes the difference between the proportion of voting power for a locality
    and the proportion of the population that town has.

    Equivalent to `compute_power_numpy` with the exception that the subsets in `subset_masks`
    includes all possible, non-empty, affirmative coalitions in this implementation.

    Note: This is slower than the `compute_power_numpy` function.

    Args:
        u_matrix (numpy.ndarray): Array of size (num_towns, 1) that contains the weights to assign
            to each town.
        subset_masks (numpy.ndarray): Array of size (num_towns, num_subsets) that contains
            boolean masks for whether a town voted in favor of a particular measure. Note:
            this should contain all possible, non-empty coalitions.
        T (float): The threshold for the power.

    Returns:
        numpy.ndarray:
            An array of size (num_towns,) that contains the computed voting power for when each
            town is given weight determined by the u_matrix.
    """
    totu = u_matrix.sum()
    T = T * totu
    a_matrix = u_matrix.T @ subset_masks

    a_minus_T = a_matrix - T

    X = subset_masks & (u_matrix >= a_minus_T) & (a_matrix > T)

    non_zero_count = np.count_nonzero(X, axis=1)
    p = non_zero_count / np.sum(non_zero_count)

    return p


def computer_power_helper_numpy_relaxed(
    m: np.ndarray,
    weights: np.ndarray,
    subset_masks: np.ndarray,
    T: float,
):
    # This computes the power discrepancy if the functions aren’t linear threshold functions, but linear bounded functions
    # as in the Diakonikolas et al. paper.
    n = len(m)

    contained_matrix = 2.0 * subset_masks - 1.0  # (num_towns, num_subsets)

    subset_weights = np.transpose(contained_matrix).dot(weights)  # num_subsets
    subset_weights_min_T = subset_weights - T
    payoffs = np.clip(subset_weights_min_T, a_min=-1, a_max=1)  # num_subsets

    banzhaf_indices = contained_matrix.dot(payoffs) / 2 ** (n - 1)

    return banzhaf_indices - m


def compute_power_numpy_simple_dyn_tau(
    u_matrix: np.ndarray,
    subset_masks: np.ndarray,
    T: float,
    numerator: int,
    denominator: int,
):
    """
    This function computes the difference between the proportion of voting power for a locality
    and the proportion of the population that town has.

    This function is modified from the `compute_power_numpy_simple` to account for the dynamic
    propensity (tau was the variable used to stand for propensity in earlier notes). The idea
    is that if the threshold needed for something to pass is tau = numerator / denominator, then
    every player's power is maximized when the expected outcome of a vote is near the threshold
    of passage.

    Args:
        u_matrix (numpy.ndarray): Array of size (num_towns, 1) that contains the weights to assign
            to each town.
        subset_masks (numpy.ndarray): Array of size (num_towns, num_subsets) that contains
            boolean masks for whether a town voted in favor of a particular measure. Note:
            this should contain all possible, non-empty coalitions.
        T (float): The threshold for passage.
        numerator (int): The numerator of the propensity fraction (tau).
        denominator (int): The denominator of the propensity fraction (tau).
    """
    totu = u_matrix.sum()
    T = T * totu
    a_matrix = u_matrix.T @ subset_masks

    a_minus_T = a_matrix - T

    X = subset_masks & (u_matrix >= a_minus_T) & (a_matrix > T)

    yes_weights = np.power(
        (numerator / (denominator - numerator)), subset_masks.sum(axis=0)
    )
    weighted_counts_matrix = X * yes_weights[np.newaxis, :]

    non_zero_count = weighted_counts_matrix.sum(axis=1)
    p = non_zero_count / np.sum(non_zero_count)

    return p
