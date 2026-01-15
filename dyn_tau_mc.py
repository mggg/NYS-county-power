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


@click.command()
@click.option(
    "--uuid",
    default=f"{int(random.random()*(10**8)):08d}",
    help="A unique integer for identifying the run",
)
@click.option("--burst-length", default=1, help="The length of the burst to run")
@click.option("--n-bursts", default=1000, help="The number of bursts to run the ")
@click.option("--threshold", default=0.5, help="The threshold for the discrep")
@click.option(
    "--show-progress",
    is_flag=True,
    help="Flag for showing the progress bar. Useful for testing",
)
@click.option("--population", default=10000, help="The population size")
@click.option(
    "--random-fuzz-epsilon",
    default=0.0,
    help="The amount of random fuzz to add to the weights",
    type=click.FloatRange(0.0, 1.0),
)
def main(
    uuid,
    burst_length,
    n_bursts,
    threshold,
    show_progress,
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

    # Fuzz the starting weights and apportion using Adams
    # We might just not do this later, but it's nice to have everyone start from the same place
    fuzzed_m = m_orig * np.random.uniform(
        1 - random_fuzz_epsilon, 1 + random_fuzz_epsilon, len(m_orig)
    )
    in_arr = appm.compute("adams", fuzzed_m, population)

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

    best_so_far = initial_discrep
    best_u = cu_u.copy()
    best_power = curr_power.copy()

    file_name = f"Ontario_DYNTAU_MCMC_discrep_L1_lenburst_{burst_length}_iters_{n_bursts}_pop_{population}_T_{threshold}_id_{uuid}_dyn_fuzz_{random_fuzz_epsilon}.jsonl"

    output_folder = Path("./MCMC_results_dyn_tau/").resolve()
    output_folder.mkdir(parents=True, exist_ok=True)

    with open(f"{output_folder}/{file_name}", "w") as f:
        print(
            f"{str('{')}\"step\": {-1}, \"discrepancy\": {initial_discrep}, \"weights\": {best_u.astype(int).tolist()}, \"power\": {best_power.astype(float).tolist()}{str('}')}",
            file=f,
            flush=True,
        )

        cur_discrep = initial_discrep
        # previous_write_idx = 0

        for i in tqdm(
            range(0, n_bursts),
            miniters=n_bursts / 1000,
            disable=not show_progress,
        ):
            # Compute the sampling probabilities
            probabilities = cp.abs(curr_diff)
            probabilities /= cp.sum(probabilities)

            indices = cp.random.choice(
                len(curr_diff), size=burst_length, p=probabilities
            )
            best_local_discrep = cur_discrep
            best_local_u = cu_u.copy()
            best_local_power = curr_power.copy()

            cu_u_temp = cu_u.copy()
            power_temp = curr_power.copy()
            discrep_temp = cur_discrep.copy()
            diff_temp = curr_diff.copy()

            for idx in indices:
                # Should we use diff_temp[idx] or curr_diff[idx]?
                # delta = 1 if diff_temp[idx] < 0 else -1
                delta = 1 if curr_diff[idx] < 0 else -1
                cu_u_temp[idx] += delta

                cu_umat_temp = cu_u_temp[:, cp.newaxis]
                power_temp = pow_fn(u_matrix=cu_umat_temp)
                diff_temp = power_temp - cu_m
                discrep_temp = discrep_function(diff_temp)

                if discrep_temp < best_local_discrep:
                    best_local_discrep = discrep_temp
                    best_local_u = cu_u_temp.copy()
                    best_local_power = power_temp.copy()

            # Apply the best single update
            if best_local_discrep < best_so_far:
                best_so_far = best_local_discrep
                best_u = best_local_u.copy()
                best_power = best_local_power.copy()
                print(
                    f"{str('{')}\"step\": {i}, \"discrepancy\": {best_so_far}, \"weights\": {best_u.astype(int).tolist()}, \"power\": {best_power.astype(float).tolist()}{str('}')}",
                    file=f,
                    flush=True,
                )
                # previous_write_idx = i
                cu_u = best_local_u
                curr_power = best_local_power
                curr_diff = curr_power - cu_m
                cur_discrep = best_local_discrep
            else:
                reject_cut = np.exp(-discrep_temp / cur_discrep)
                if np.random.rand() < reject_cut:
                    cu_u = cu_u_temp
                    curr_power = power_temp
                    curr_diff = curr_power - cu_m
                    cur_discrep = discrep_temp

            # if i - previous_write_idx > 100 * burst_length:
            #     burst_length = min(100, burst_length * 2)


if __name__ == "__main__":
    main()
