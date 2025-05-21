#!/bin/bash
#SBATCH --account=p904-24-3
#SBATCH --mail-user=<jakub.kopal@kinit.sk>

## Nodes allocation
#SBATCH --partition=gpu
#SBATCH --nodes=1 
#SBATCH --cpus-per-task=16
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
set -xe


### SBATCH --time=20:00:00 # Estimate to increase job priority

eval "$(conda shell.bash hook)"
conda activate imagenet

python main.py --job_name overshoot --optimizer_name sgdo --overshoot 5 --seed 11 -a resnet50 imagenet

