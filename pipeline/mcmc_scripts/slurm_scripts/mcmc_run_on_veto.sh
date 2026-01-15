#!/bin/bash 
#SBATCH --job-name=NYS_queries_on_veto
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --time=3-00:00:00
#SBATCH --output=MCMC_logs/%x_%j.out

threshold=$1
nbursts=$2
uuid=$3
fuzz=$4
pop=$5

python mc_ontario_veto.py \
    --threshold $threshold \
    --n-bursts $nbursts \
    --burst-length 5 \
    --uuid $uuid \
    --random-fuzz-epsilon $fuzz \
    --population $pop

