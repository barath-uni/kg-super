#!/bin/bash

#SBATCH --partition=gpu
#SBATCH --gpus-per-node=1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=18
#SBATCH --job-name=SIMKGCRelEval
#SBATCH --time=00:20:00
#SBATCH --mem=100G
#SBATCH --output=simkgcrelationshipeval.out

module purge
module load 2021
module load Anaconda3/2021.05

# /home/barath/blogwriter/AutoBlogWriter
cd $HOME/kg-super-engine/kg-super-engine/simkgc/SimKGC
eval "$(conda shell.bash hook)"
conda activate kg-rmpi2

# Not adding a negative sampling until I figure out what it does
python3 -u evaluate.py --task FB15K237  --is-test  --eval-model-path "$HOME/kg-super-engine/kg-super-engine/simkgc/SimKGC/relationship/model_best.mdl"  --neighbor-weight 0.05  --rerank-n-hop 2 --train-path "$HOME/kg-super-engine/kg-super-engine/simkgc/SimKGC/data/FB15k237/train.txt.json" --valid-path "$HOME/kg-super-engine/kg-super-engine/simkgc/SimKGC/data/FB15k237/valid.txt.json"