#!/bin/bash
#SBATCH --job-name=dimresnet
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --hint=nomultithread
#SBATCH --time=00:20:00
##SBATCH --time=19:59:00
#SBATCH --output=dimresnet-%j.out
#SBATCH --error=dimresnet-%j.out
#SBATCH --exclusive
#SBATCH -C h100
#SBATCH --cpus-per-task=24 # Pour dalia chaque GPU a 36
#SBATCH --qos=qos_gpu_h100-dev

# Nettoyage des modules charges en interactif et herites par defaut
module purge

# Chargement des modules
module load arch/h100
module load pytorch-gpu/py3/2.7.0

# Echo des commandes lancees
set -x

#export CUBLAS_WORKSPACE_CONFIG=:4096:8
# Execution du code
srun python dimresnet.py -b 64 --lr 0.8 -e 1 -n 24 --drop-last
