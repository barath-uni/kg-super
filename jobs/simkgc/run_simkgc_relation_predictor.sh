#!/bin/bash

#SBATCH --partition=gpu
#SBATCH --gpus-per-node=1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=18
#SBATCH --job-name=SIMKGCReltrain
#SBATCH --time=12:00:00
#SBATCH --mem=120G
#SBATCH --output=simkgcrelationswap.out

module purge
module load 2021
module load Anaconda3/2021.05

# /home/barath/blogwriter/AutoBlogWriter
cd $HOME/kg-super-engine/kg-super-engine/simkgc/SimKGC
eval "$(conda shell.bash hook)"
conda activate kg-rmpi2

python3 -u main.py --model-dir "$HOME/kg-super-engine/kg-super-engine/simkgc/SimKGC/train" --pretrained-model bert-base-uncased --pooling mean --lr 1e-5 --use-link-graph --train-path "$HOME/kg-super-engine/kg-super-engine/simkgc/SimKGC/data/FB15k237/train.txt.json" --valid-path "$HOME/kg-super-engine/kg-super-engine/simkgc/SimKGC/data/FB15k237/valid.txt.json" --task FB15k237 --batch-size 512 --print-freq 20 --additive-margin 0.02 --use-amp --use-self-negative --finetune-t --pre-batch 2 --epochs 10 --workers 4 --max-to-keep 5