# NYS County Power

This repository contains code for computing the voting power of a given county in NYS relative
to the population proportion of that county. Ideally the difference between the voting power
of a county and the population proportion of that county should be 0 (i.e. it should adhere to the
principle of one person, one vote).


## Setup 

You can set up your conda environment by running the following command:

```bash
conda env create -f conda_environment.yml
conda activate nys
```

and then you will need to check which version of CUDA is installed on your machine using the 
command `nvcc --version`.

```console
(nys) 12:57:10 ❯ nvcc --version

nvcc: NVIDIA (R) Cuda compiler driver
Copyright (c) 2005-2025 NVIDIA Corporation
Built on Fri_Nov__7_07:23:37_PM_PST_2025
Cuda compilation tools, release 13.1, V13.1.80
Build cuda_13.1.r13.1/compiler.36836380_0
```

Here we can see that the CUDA version is 13.1, which is what is set up in the UV environment.
If you have a different version of CUDA installed (probably 12), you will need to modify the 
pyproject.toml file using the command

```bash
uv add cupy-cuda12x
uv sync
```

If you have CUDA version 13 installed, you can set up the environment using the command

```bash
uv sync --locked
```

After that, you can test that everything is working by running 

```bash
uv run pipeline/mcmc_scripts/python_scripts/mc_ontario.py --burst-length 5 --n-bursts 100 --threshold 0.5 --show-progress --population 100000
```
