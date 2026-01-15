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
        4625364,
        672591,
        6547629,
        2853118,
        37253956,
        5303925,
        3574097,
        897934,
        625741,
        19378102,
        9883640,
        1360301,
        1826341,
        12702379,
        6346105,
        2967297,
        2700551,
        4339367,
        4533372,
        1316470,
        5988927,
        6483802,
        9535483,
        5029196,
        2763885,
        5773552,
        989415,
        1852994,
        2915918,
        1328361,
        8791894,
        2059179,
        18801310,
        9687653,
        710231,
        11536504,
        3751351,
        3831074,
        12830632,
        1052567,
        4779736,
        814180,
        6392017,
        25145561,
        3046355,
        601723,
        8001024,
        6724540,
        1567582,
        5686986,
        563626,
    ]
)

u = np.array(
    [
        9,
        3,
        11,
        6,
        55,
        9,
        7,
        3,
        3,
        29,
        16,
        4,
        4,
        20,
        11,
        6,
        6,
        8,
        8,
        4,
        10,
        11,
        16,
        10,
        6,
        10,
        3,
        5,
        6,
        4,
        14,
        5,
        29,
        15,
        3,
        18,
        7,
        7,
        20,
        4,
        9,
        3,
        11,
        38,
        6,
        3,
        13,
        12,
        5,
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
).to_csv(f"power_2010_samples_{sample_size*n_samples}.csv", index=False)
