import itertools
import numpy as np
from functools import partial
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
TOP_DIR = SCRIPT_DIR.parents[0]

import sys

sys.path.append(str(TOP_DIR / "power_functions"))

from numpy_power_function import compute_power_numpy_simple_dyn_tau as compute_power

m = [1, 2, 2, 3]
m_orig = m.copy()
numtowns = len(m)
A = set(range(numtowns))
A_subsets = []

# Note: We assume that one town, in particular, always votes in the affirmative,
for i in range(len(A) + 1):
    for subset in itertools.combinations(A, i):
        A_subsets.append(list(set(subset)))


subset_masks = np.zeros((len(m_orig), len(A_subsets)), dtype=np.float64)
for j, subset in enumerate(A_subsets):
    subset_masks[subset, j] = True

subset_masks_bool = subset_masks.astype(np.bool_)
subset_masks_float = subset_masks.astype(np.float64)

print("Subset masks boolean: ", subset_masks_float.T)

in_arr = m.copy()


threshold = 2 / 3

pow_fn = partial(
    compute_power,
    subset_masks=subset_masks_bool,
    T=threshold,
    numerator=2,
    denominator=3,
)

u_mat = np.asarray(in_arr)[:, np.newaxis]

curr_power = pow_fn(u_matrix=u_mat)
print("Current power:", curr_power)
