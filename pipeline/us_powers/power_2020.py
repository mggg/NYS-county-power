import numpy as np
from tqdm import tqdm

names = np.array(
    [
        "AL",
        "AK",
        "AZ",
        "AR",
        "CA",
        "CO",
        "CT",
        "DE",
        "DC",
        "FL",
        "GA",
        "HI",
        "ID",
        "IL",
        "IN",
        "IA",
        "KS",
        "KY",
        "LA",
        "ME",
        "MD",
        "MA",
        "MI",
        "MN",
        "MS",
        "MO",
        "MT",
        "NE",
        "NV",
        "NH",
        "NJ",
        "NM",
        "NY",
        "NC",
        "ND",
        "OH",
        "OK",
        "OR",
        "PA",
        "RI",
        "SC",
        "SD",
        "TN",
        "TX",
        "UT",
        "VT",
        "VA",
        "WA",
        "WV",
        "WI",
        "WY",
    ]
)

m = np.array(
    [
        5024279,
        733391,
        7151502,
        3011524,
        39538223,
        5773714,
        3605944,
        689545,
        989948,
        21538187,
        10711908,
        1455271,
        1839106,
        12812508,
        6785528,
        3190369,
        2937880,
        4505836,
        4657757,
        1362359,
        6177224,
        7029917,
        10077331,
        5706494,
        2961279,
        6154913,
        1084225,
        1961504,
        3104614,
        1377529,
        9288994,
        2117522,
        20201249,
        10439388,
        779094,
        11799448,
        3959353,
        4237256,
        13002700,
        1097379,
        5118425,
        886667,
        6910840,
        29145505,
        3271616,
        643077,
        8631393,
        7705281,
        1793716,
        5893718,
        576851,
    ]
)

u = np.array(
    [
        9,
        3,
        11,
        6,
        54,
        10,
        7,
        3,
        3,
        30,
        16,
        4,
        4,
        19,
        11,
        6,
        6,
        8,
        8,
        4,
        10,
        11,
        15,
        10,
        6,
        10,
        4,
        5,
        6,
        4,
        14,
        5,
        28,
        16,
        3,
        17,
        7,
        8,
        19,
        4,
        9,
        3,
        11,
        40,
        6,
        3,
        13,
        12,
        4,
        10,
        3,
    ]
)


sort_ind = np.argsort(m)[::-1]
names = names[sort_ind]
m = m[sort_ind]
u = u[sort_ind]


numtowns = len(m)
A = set(range(numtowns))
A_subsets = []

sample_size = 10_000_000
n_samples = 1000


non_zero_count = np.zeros(numtowns)
n_appeared = np.zeros(numtowns)
tot_u = u.sum()
T = (1 / 2) * tot_u
u_mat = u[:, np.newaxis]

for i in tqdm(range(n_samples)):
    A_subsets = np.random.choice([True, False], size=(numtowns, sample_size))

    a_matrix = u @ A_subsets
    a_minus_T = a_matrix - T

    X = A_subsets & (u_mat >= a_minus_T) & (a_matrix > T)

    non_zero_count += np.count_nonzero(X, axis=1)
    n_appeared += np.count_nonzero(A_subsets, axis=1)

p = non_zero_count / non_zero_count.sum()

banz_prob = non_zero_count / (2 * n_appeared)

for item in p:
    print(f"{np.around(item,4):.4f}")

print("=" * 30)

for item in banz_prob:
    print(f"{np.around(item,4):.4f}")

import pandas as pd

pd.DataFrame(
    columns=[
        "name",
        "population",
        "weight",
        "non_zero_count",
        "n_appeared",
        "banz_prob",
        "power",
    ],
    data=np.column_stack((names, m, u, non_zero_count, n_appeared, banz_prob, p)),
).to_csv(f"power_2020_samples_{sample_size*n_samples}.csv", index=False)
