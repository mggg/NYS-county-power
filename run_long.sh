#!/bin/bash

thresholds=(0.5 0.67 0.75 0.8)
n_iters=(10000 10000 100000 100000)
n_iters_random=(10000 10000 100000 100000)

thresholds=(0.8)
n_iters=(10000)
n_iters_random=(10000)

# for i in "${!thresholds[@]}"
# do
#     threshold=${thresholds[$i]}
#     python mc_livingston.py --threshold $threshold --n-bursts ${n_iters[$i]} --burst-length 5 --show-progress --uuid 43 
#     # python mc_liv_take2.py --threshold $threshold --n-bursts ${n_iters[$i]} --burst-length 5 --show-progress --uuid 43 --population 100000
#     python mc_livingston.py --threshold $threshold --n-bursts ${n_iters_random[$i]} --burst-length 5 --show-progress --uuid 43 --random-fuzz-epsilon 0.8
# done

for i in "${!thresholds[@]}"
do
    threshold=${thresholds[$i]}
    python mc_ontario.py --threshold $threshold --n-bursts ${n_iters[$i]} --burst-length 5 --show-progress --uuid 43 
    python mc_ontario.py --threshold $threshold --n-bursts ${n_iters_random[$i]} --burst-length 5 --show-progress --uuid 43 --random-fuzz-epsilon 0.8
done

# for threshold in "${thresholds[@]}"
# do
#     python mc_on_take2.py --threshold $threshold --n-bursts $n_iters --burst-length 5 --show-progress --uuid 42
# done

# for threshold in "${thresholds[@]}"
# do
#     python mc_ontario.py --threshold $threshold --n-bursts $n_iters --burst-length 3 --show-progress
# done
