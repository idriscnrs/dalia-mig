# Benchmark pour mig

C'est un entrainement d'un resnet50 sur imagenet

Besoin du dataset imagenet disponible https://www.image-net.org/download-images
Ce dataset est aussi disponible sur le DSDIR de jean-zay
Il faut adapter les lignes 32 à 34 dans le fichier dimresnet.py pour pointer sur
le bon repertoire du dataset imagenet.

# Pour lancer le job

daliamig1.slurm un exemple de script qui tourne sur 1 GPU

daliamig2.slurm un exemple qui tournerait sur un noeud entier avec 7 MIG sur les
4 GPUs du noeud et donc 28 gpu/tasks = 4(gpu/node)*7(mig)

# Contenu

bind_gpu.sh : Script pour faire le binding gpu correctement pour daliamig2.slurm
daliamig1.slurm : Exemple de soumission d'un travail utilisant 1 GPU
daliamig2.slurm : Exemple de soumission d'un meme travail sur tout les GPUs d'un noeud
dimreset.py : Script Python d'apprentissage d'un resnet50
zay : Répertoire avec les scripts et les sorties pour jean-zay h100