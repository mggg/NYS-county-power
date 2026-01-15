import pandas as pd
import jsonlines as jl
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colormaps as mcm

from glob import glob

all_files = glob("../experimental_MCMC_results/*.jsonl")


import itertools

all_settings = itertools.product(
    [0.5, 2 / 3, 0.75], [10_000, 100_00], ["L1", "log"], ["Livingston", "Ontario"]
)

data_folder = "../experimental_MCMC_results"
figure_folder = "../figures"

for T, n_iters, method, town in all_settings:
    file_name = f"{data_folder}/{town}_MCMC_discrep_{method}_lenburst_1_iters_{n_iters}_T_{T}_id_66003712.jsonl"

    try:
        with open(file_name, "r") as f:
            pass
    except:
        continue

    disc_array = []
    disc_steps = []
    with jl.open(file_name, "r") as f:
        initial_step = f.read()
        initial_prop = np.array(initial_step["weights"])
        initial_prop = initial_prop / np.sum(initial_prop)
        disc_steps.append(0)
        disc_array.append((initial_prop - initial_prop) / initial_prop)
        for line in f:
            disc_steps.append(line["step"])
            line_disc = np.array(line["weights"])
            line_disc = line_disc / np.sum(line_disc)
            disc_array.append((line_disc - initial_prop) / initial_prop)

    disc_array_np = np.array(disc_array).T

    color_list = mcm.get_cmap("viridis")(np.linspace(0, 1, disc_array_np.shape[0]))

    fig, ax = plt.subplots(figsize=(15, 10))

    for i, row in enumerate(disc_array_np):
        ax.plot(row, color=color_list[i], label=f"T={row[0]}")

    ax.set_xticks([])
    plt.tight_layout()
    plt.savefig(
        f"{figure_folder}/{'Ontario' if disc_array_np.shape[0] == 21 else 'Livingston'}_(wt[i]-m[i])_div_(m[i])_for_threshold_{T}_method_{method}_steps_{n_iters}.png"
    )
    plt.close()

    disc_log_array = []
    disc_log_steps = []
    with jl.open(file_name, "r") as f:
        initial_log_step = f.read()
        initial_log_prop = np.array(initial_log_step["weights"])
        initial_log_power = np.array(initial_log_step["power"])
        initial_log_prop = initial_log_prop / np.sum(initial_log_prop)
        disc_log_steps.append(0)
        disc_log_array.append(np.log(initial_log_power / initial_log_prop))
        for line_log in f:
            disc_log_steps.append(line_log["step"])
            line_log_disc = np.array(line_log["power"])
            # line_log_disc = line_log_disc / np.sum(line_log_disc)
            disc_log_array.append(np.log(line_log_disc / initial_log_prop))

    disc_log_array_np = np.array(disc_log_array).T

    fig, ax = plt.subplots(figsize=(15, 10))

    for i, row in enumerate(disc_log_array_np):
        ax.plot(row, color=color_list[i], label=f"T={row[0]}")

    ax.set_xticks([])
    plt.tight_layout()
    plt.savefig(
        f"{figure_folder}/{'Ontario' if disc_array_np.shape[0] == 21 else 'Livingston'}_log(p[i]_div_m[i])_for_threshold_{T}_method_{method}_steps_{n_iters}.png"
    )

    plt.close()
