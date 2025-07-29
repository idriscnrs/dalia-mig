#!/bin/bash
#SBATCH --job-name=dimresnet
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=28 # 7(mig)*4 GPU
#SBATCH --gres=gpu:28 # 28 gpu ?
#SBATCH --hint=nomultithread
#SBATCH --time=00:20:00
#SBATCH --output=dimresnet-%j.out
#SBATCH --error=dimresnet-%j.out
#SBATCH --exclusive
#SBATCH --cpus-per-task=5 # Pour dalia en mig 24g ca donne 5 cpu/gpu

# Nettoyage des modules charges en interactif et herites par defaut
module purge

# Chargement des modules
#module load pytorch-gpu/py3/2.7.0

# Echo des commandes lancees
set -x

# Execution du code
# bind_gpu.sh est un script pour definir le CUDA_VISIBLE_DEVICES different pour
# chaque gpu
srun ./bind_gpu.sh python dimresnet.py -b 64 --lr 0.8 -e 1 -n 5 --drop-last
