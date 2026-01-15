#!/usr/bin/env bash

thresholds=(0.5 0.67 0.75 0.8)
thresholds=(0.6)
# seeds=(3721 17472 28940 7909 14132) # 28940 causes issues?
seeds=(3721 17472 195 7909 14132)
# seeds=(195)

niters=100000
startingpop=100000

for i in "${!thresholds[@]}"
do
for seed in "${seeds[@]}"
do
for fuzz in 0.0 0.8
do
    # sbatch mcmc_run_liv_dyn_tau.sh ${thresholds[$i]} $niters $seed $fuzz $startingpop
    sbatch mcmc_run_on_dyn_tau.sh ${thresholds[$i]} $niters $seed $fuzz $startingpop
done
done
done
