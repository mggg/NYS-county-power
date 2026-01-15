import numpy as np
from tqdm import tqdm

names = np.array(
    [
        "TX",
        "NY",
        "FL",
        "CA",
        "IL",
        "PA",
        "MI",
        "OH",
        "NJ",
        "GA",
        "IN",
        "VA",
        "NC",
        "WA",
        "WI",
        "MA",
        "MD",
        "TN",
        "AZ",
        "KY",
        "MO",
        "SC",
        "OK",
        "LA",
        "OR",
        "AL",
        "CT",
        "MN",
        "MS",
        "CO",
        "UT",
        "KS",
        "AR",
        "NV",
        "MT",
        "IA",
        "NM",
        "WV",
        "ID",
        "DE",
        "ME",
        "SD",
        "NH",
        "HI",
        "NE",
        "ND",
        "AK",
        "VT",
        "DC",
        "RI",
        "WY",
    ]
)

m = np.array(
    [
        20851820,
        18976457,
        15982378,
        33871648,
        12419293,
        12281054,
        9938444,
        11353140,
        8414350,
        8186453,
        6080485,
        7078515,
        8049313,
        5894121,
        5363675,
        6349097,
        5296486,
        5689283,
        5130632,
        4041769,
        5595211,
        4012012,
        3450654,
        4468976,
        3421399,
        4447100,
        3405565,
        4919479,
        2844658,
        4301261,
        2233169,
        2688418,
        2673400,
        1998257,
        902195,
        2926324,
        1819046,
        1808344,
        1293953,
        783600,
        1274923,
        754844,
        1235786,
        1211537,
        1711263,
        642200,
        626932,
        608827,
        572059,
        1048319,
        493782,
    ]
)

u = np.array(
    [
        34,
        31,
        27,
        55,
        21,
        21,
        17,
        20,
        15,
        15,
        11,
        13,
        15,
        11,
        10,
        12,
        10,
        11,
        10,
        8,
        11,
        8,
        7,
        9,
        7,
        9,
        7,
        10,
        6,
        9,
        5,
        6,
        6,
        5,
        3,
        7,
        5,
        5,
        4,
        3,
        4,
        3,
        4,
        4,
        5,
        3,
        3,
        3,
        3,
        4,
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
).to_csv(f"power_2000_samples_{sample_size*n_samples}.csv", index=False)
