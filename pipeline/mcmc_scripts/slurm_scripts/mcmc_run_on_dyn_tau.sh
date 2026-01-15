#!/bin/bash 
#SBATCH --job-name=NYS_queries_on_dyn_tau_dec25
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --time=03:00:00
#SBATCH --output=MCMC_logs/dyn_tau/%x_%j.out
#SBATCH --requeue

threshold=$1
nbursts=$2
uuid=$3
fuzz=$4
pop=$5

uv run mc_ontario_dyn_tau.py \
    --threshold $threshold \
    --n-bursts $nbursts \
    --burst-length 5 \
    --uuid $uuid \
    --random-fuzz-epsilon $fuzz \
    --population $pop

