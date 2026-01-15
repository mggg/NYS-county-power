import itertools
import numpy as np
from functools import partial
import cupy as cp
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
TOP_DIR = SCRIPT_DIR.parents[0]
print(TOP_DIR)

import sys

sys.path.append(str(TOP_DIR / "power_functions"))

from cuda_power_function import compute_power_cupy_simple_dyn_tau as compute_power
from cuda_power_function import compute_power_cupy_simple as compute_power_simple

m = [1, 2, 2, 3]
m_orig = m.copy()
numtowns = len(m)
A = set(range(numtowns))
A_subsets = []

# Note: We assume that one town, in particular, always votes in the affirmative,
for i in range(len(A) + 1):
    for subset in itertools.combinations(A, i):
        if 1 in subset:
            A_subsets.append(list(set(subset)))


subset_masks = np.zeros((len(m_orig), len(A_subsets)), dtype=np.float64)
for j, subset in enumerate(A_subsets):
    subset_masks[subset, j] = True

subset_masks_bool = subset_masks.astype(np.bool_)
subset_masks_float = subset_masks.astype(np.float64)

print("Subset masks boolean: ", subset_masks_float.T)

in_arr = m.copy()

cu_m = cp.asarray(m)
cu_u = cp.asarray(in_arr)
cu_umat = cp.asarray(in_arr)[:, cp.newaxis]
cu_subset_masks_bool = cp.asarray(subset_masks_bool)
cu_subset_masks_float = cp.asarray(subset_masks_float)


threshold = 2 / 3

pow_fn = partial(
    compute_power,
    subset_masks_bool=cu_subset_masks_bool,
    subset_masks_float=cu_subset_masks_float,
    T=threshold,
    numerator=2,
    denominator=3,
)

curr_power = pow_fn(u_matrix=cu_umat)
print("Current power:", curr_power)
