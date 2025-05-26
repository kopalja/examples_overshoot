#!/bin/bash
set -xe



SEEDS=(10 20 30 40 50 60 70 80 90 100)
for seed in ${SEEDS}; do
    python main.py --seed ${seed} --job_name vae_full_run --optimizer_name adam
    python main.py --seed ${seed} --job_name vae_full_run --optimizer_name adamo
done


