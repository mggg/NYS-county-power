import numpy as np
import pickle
from tqdm import tqdm
from functools import partial
import click
import random
import apportionment.methods as appm
import cupy as cp

from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
TOP_DIR = SCRIPT_DIR.parents[2]
PKL_DIR = TOP_DIR / "data"
OUTPUT_DIR = TOP_DIR / "MCMC_results_veto"
OUTPUT_DIR.mkdir(exist_ok=True)

import sys

sys.path.append(str(TOP_DIR / "power_functions"))

from cuda_power_function import compute_power_cupy as compute_power


@click.command()
@click.option(
    "--uuid",
    default=f"{int(random.random()*(10**8)):08d}",
    help="A unique integer for identifying the run",
)
@click.option("--burst-length", default=5, help="The length of the burst to run")
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
        [6945.,4158,2322,10242,1464,2087,4156,7508,4452,5341,2695,725,765,1583,2292,1157,3187,]
    ) 
    m_orig.sort()
    m = m_orig / np.sum(m_orig)  
    # fmt: on

    n_towns = len(m_orig)

    if threshold == 0.67:
        threshold = 2.0 / 3

    # Note that NYS_counties does not cover all possible subsets. There is an assumption of
    # one town always voting in the affirmative. This substantially reduces the size of the
    # matrix that
    with open(PKL_DIR / "NYS_counties_livingston.pkl", "rb") as f:
        A_subsets = pickle.load(f)

    subset_masks = np.zeros((n_towns, len(A_subsets)), dtype=np.float64)
    for j, subset in enumerate(A_subsets):
        subset_masks[subset, j] = True

    subset_masks_bool = subset_masks.astype(np.bool_)
    subset_masks_float = subset_masks.astype(np.float64)

    # Fuzz the starting weights and apportion using Adams
    # We might just not do this later, but it's nice to have everyone start from the same place
    fuzzed_m = m_orig * np.random.uniform(
        1 - random_fuzz_epsilon, 1 + random_fuzz_epsilon, n_towns
    )
    in_arr = appm.compute("adams", fuzzed_m, population)
    while any(np.array(in_arr) >= threshold * population):
        fuzzed_m = m_orig * np.random.uniform(
            1 - random_fuzz_epsilon, 1 + random_fuzz_epsilon, n_towns
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
    )

    discrep_function = lambda x: cp.sum(cp.abs(x))

    curr_power = pow_fn(u_matrix=cu_umat)
    curr_diff = curr_power - cu_m
    initial_discrep = discrep_function(curr_diff)

    best_so_far = initial_discrep
    best_u = cu_u.copy()
    best_power = curr_power.copy()

    file_name = f"Livingston_MCMCVETO_discrep_L1_lenburst_{burst_length}_iters_{n_bursts}_pop_{population}_T_{threshold}_id_{uuid}_dyn_fuzz_{random_fuzz_epsilon}.jsonl"

    with open(OUTPUT_DIR / file_name, "w") as f:
        print(
            f"{str('{')}\"step\": {-1}, \"discrepancy\": {initial_discrep}, \"weights\": {best_u.astype(int).tolist()}, \"power\": {best_power.astype(float).tolist()}{str('}')}",
            file=f,
            flush=True,
        )

        cur_discrep = initial_discrep
        n_samples = 0
        n_steps = 0
        previous_write_idx = 0
        previous_burst_idx = 0

        for i in tqdm(
            range(0, n_bursts),
            miniters=n_bursts / 1000,
            disable=not show_progress,
        ):
            # Compute the sampling probabilities
            probabilities = cp.abs(curr_diff)
            probabilities /= cp.sum(probabilities)

            best_local_discrep = cur_discrep
            best_local_u = cu_u.copy()
            best_local_power = curr_power.copy()

            cu_u_temp = cu_u.copy()
            power_temp = curr_power.copy()
            discrep_temp = cur_discrep.copy()
            diff_temp = curr_diff.copy()

            best_substep = 0
            for j in range(1, burst_length + 1):
                total_wt = cu_u_temp.sum()

                idx = cp.random.choice(n_towns, size=1, p=probabilities)[0]
                delta = 1 if diff_temp[idx] < 0 else -1
                cu_u_temp[idx] += delta
                # Only accept non-veto steps
                while any(cu_u_temp >= (total_wt + delta) * (1 - threshold)):
                    cu_u_temp[idx] -= delta
                    idx = cp.random.choice(n_towns, size=1, p=probabilities)[0]
                    delta = 1 if curr_diff[idx] < 0 else -1
                    cu_u_temp[idx] += delta

                n_samples += 1
                cu_umat_temp = cu_u_temp[:, cp.newaxis]
                power_temp = pow_fn(u_matrix=cu_umat_temp)

                diff_temp = power_temp - cu_m
                discrep_temp = discrep_function(diff_temp)

                if discrep_temp < best_local_discrep:
                    best_local_discrep = discrep_temp
                    best_local_u = cu_u_temp.copy()
                    best_local_power = power_temp.copy()
                    best_substep = j

            # Apply the best single update
            if best_local_discrep < best_so_far:
                best_so_far = best_local_discrep
                best_u = best_local_u.copy()
                best_power = best_local_power.copy()
                n_steps += best_substep
                assert not any(
                    best_u >= best_u.sum() * (1 - threshold)
                ), "BAD best_u allowed"
                print(
                    (
                        f"{str('{')}\"step\": {i}, "
                        f'"discrepancy": {best_so_far}, '
                        f'"weights": {best_u.astype(int).tolist()}, '
                        f'"power": {best_power.astype(float).tolist()}, '
                        f'"burst_length": {burst_length}, '
                        f'"n_steps": {n_steps}, '
                        f"\"n_samples\": {n_samples}{str('}')}"
                    ),
                    file=f,
                    flush=True,
                )
                previous_write_idx = i
                cu_u = best_local_u
                curr_power = best_local_power
                curr_diff = curr_power - cu_m
                cur_discrep = best_local_discrep
            else:
                reject_cut = np.exp(-discrep_temp / (2 * cur_discrep))
                if np.random.rand() < reject_cut:
                    cu_u = cu_u_temp
                    curr_power = power_temp
                    curr_diff = curr_power - cu_m
                    cur_discrep = discrep_temp
                    n_steps += burst_length

            assert not any(cu_u >= cu_u.sum() * (1 - threshold)), "BAD cu_u allowed"
            if i - previous_write_idx > 50 and i - previous_burst_idx > 50:
                burst_length = min(int(0.10 * (cu_u.sum().item())), burst_length * 10)
                previous_burst_idx = i

        print(
            (
                f"{str('{')}\"step\": {n_bursts}, "
                f'"discrepancy": {best_so_far}, '
                f'"weights": {best_u.astype(int).tolist()}, '
                f'"power": {best_power.astype(float).tolist()}, '
                f'"burst_length": {burst_length}, '
                f'"n_steps": {n_steps}, '
                f"\"n_samples\": {n_samples}{str('}')}"
            ),
            file=f,
            flush=True,
        )


if __name__ == "__main__":
    main()
