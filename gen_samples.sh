#!/bin/sh
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --time=40:00:00
#SBATCH --gres=gpu:a100:2
#SBATCH --constraint=a100-80gb
#SBATCH --mem-per-gpu=40GB
#SBATCH --cpus-per-gpu=20
#SBATCH --partition=gpu

pip install -e .
aria sample -m large -c large-abs-inst.safetensors -p tests/test_data/bach.mid tests/test_data/arabseque.mid -var 10 -trunc 0 -l 1600 -temp 0.95 -form ABA 