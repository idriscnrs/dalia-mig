#!/bin/bash
#SBATCH --job-name=dimresnet
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --hint=nomultithread
#SBATCH --time=00:20:00
#SBATCH --output=dimresnet-%j.out
#SBATCH --error=dimresnet-%j.out
#SBATCH --exclusive
#SBATCH --cpus-per-task=5

# Nettoyage des modules charges en interactif et herites par defaut
module purge

# Chargement des modules
#module load pytorch-gpu/py3/2.7.0

# Echo des commandes lancees
set -x

# Execution du code
# Batch size 64, learning rate 0.8, 1 epochs, 5 workers, drop last batch
srun python dimresnet.py -b 64 --lr 0.8 -e 1 -n 5 --drop-last
