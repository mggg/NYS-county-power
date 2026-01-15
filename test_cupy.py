import numpy as np
import pickle
from tqdm import tqdm
from functools import partial
import click
import random
import apportionment.methods as appm
from fractions import Fraction
from pathlib import Path

import cupy as cp

from cuda_power_function import compute_power_cupy_simple_dyn_tau as compute_power


def main(
    uuid,
    threshold,
    population,
    random_fuzz_epsilon,
):
    cp.random.seed(seed=int(uuid))
    np.random.seed(seed=int(uuid))

    # fmt: off
    m_orig = np.array(
        [2284,1668,5436,5140,11109, 3640,14170,5210,3921,3679, 3473,4106,3931,9404,2403, 6637,3360,2644,1641,15860, 2740]
    )
    m_orig.sort()
    m = m_orig / np.sum(m_orig)  
    # fmt: on

    if threshold == 0.67:
        threshold = 2.0 / 3

    fraction_threshold = Fraction(threshold).limit_denominator()
    numerator = fraction_threshold.numerator
    denominator = fraction_threshold.denominator
    print(numerator, denominator)

    with open("NYS_counties_ontario_full.pkl", "rb") as f:
        A_subsets = pickle.load(f)

    subset_masks = np.zeros((len(m_orig), len(A_subsets)), dtype=np.float64)
    for j, subset in enumerate(A_subsets):
        subset_masks[subset, j] = True

    subset_masks_bool = subset_masks.astype(np.bool_)
    subset_masks_float = subset_masks.astype(np.float64)
    base = [
        1517,
        1540,
        2090,
        2202,
        2427,
        2519,
        3052,
        3164,
        3324,
        3361,
        3585,
        3594,
        3755,
        4679,
        4743,
        4947,
        6013,
        8363,
        9770,
        12100,
        13255,
    ]

    base = [
        1488,
        1511,
        2106,
        2211,
        2421,
        2506,
        3123,
        3204,
        3340,
        3373,
        3582,
        3589,
        3740,
        4682,
        4740,
        4932,
        6000,
        8381,
        9748,
        12092,
        13231,
    ]
    base = [
        1431,
        1445,
        2169,
        2248,
        2415,
        2483,
        3226,
        3289,
        3392,
        3418,
        3580,
        3584,
        3705,
        4698,
        4742,
        4895,
        5990,
        8334,
        9700,
        12032,
        13224,
    ]
    base = [
        1286,
        1288,
        2305,
        2333,
        2395,
        2420,
        3440,
        3467,
        3508,
        3513,
        3575,
        3577,
        3619,
        4735,
        4750,
        4805,
        5961,
        8334,
        9590,
        11940,
        13159,
    ]
    # base = [
    #     1522,
    #     1547,
    #     2098,
    #     2211,
    #     2437,
    #     2528,
    #     3063,
    #     3177,
    #     3338,
    #     3375,
    #     3599,
    #     3608,
    #     3770,
    #     4698,
    #     4762,
    #     4967,
    #     6037,
    #     8396,
    #     9809,
    #     12149,
    #     13308,
    # ]
    print("Base apportionment:", base)
    print("Base apportionment sum:", sum(base))
    new = appm.compute("huntington", base, population)
    print(new)
    print(sum(new))
    in_arr = np.array(new)
    cu_m = cp.asarray(m)
    cu_u = cp.asarray(in_arr)
    cu_umat = cp.asarray(in_arr)[:, cp.newaxis]
    cu_subset_masks_bool = cp.asarray(subset_masks_bool)
    cu_subset_masks_float = cp.asarray(subset_masks_float)

    pow_fn = partial(
        compute_power,
        subset_masks_bool=cu_subset_masks_bool,
        subset_masks_float=cu_subset_masks_float,
        T=threshold,
        numerator=numerator,
        denominator=denominator,
    )

    discrep_function = lambda x: cp.sum(cp.abs(x))

    curr_power = pow_fn(u_matrix=cu_umat)
    curr_diff = curr_power - cu_m
    initial_discrep = discrep_function(curr_diff)

    print(f"Initial discrepancy: {initial_discrep.get():.12f}")
    print(f"Initial power: {curr_power.get()}")


if __name__ == "__main__":
    main(42, 3 / 4, 100000, 0.0)
