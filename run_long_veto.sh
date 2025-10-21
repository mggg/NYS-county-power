#!/bin/bash

thresholds=(0.5 0.67 0.75 0.8)
seeds=(3721 17472 28940)

# thresholds=(0.5 0.8)
# seeds=(2025)
niters=100000
startingpop=100000

for i in "${!thresholds[@]}"
do
for seed in "${seeds[@]}"
do
    sbatch mcmc_run_liv_veto.sh ${thresholds[$i]} $niters $seed 0.0 $startingpop
    sbatch mcmc_run_liv_veto.sh ${thresholds[$i]} $niters $seed 0.8 $startingpop
    sbatch mcmc_run_on_veto.sh ${thresholds[$i]} $niters $seed 0.0 $startingpop
    sbatch mcmc_run_on_veto.sh ${thresholds[$i]} $niters $seed 0.8 $startingpop
done
done
