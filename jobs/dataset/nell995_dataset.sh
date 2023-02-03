#!/bin/bash

#SBATCH --partition=gpu
#SBATCH --gpus-per-node=1
#SBATCH --job-name=Trainmetatitan
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=3
#SBATCH --time=1:30:00
#SBATCH --mem=32000M
#SBATCH --output=nell995.out

module purge
module load 2021
module load Anaconda3/2021.05

cd $HOME/kg-super-engine/kg-super-engine
eval "$(conda shell.bash hook)"
conda activate kg-super

mkdir -p /home/barath/kg-super-engine/kg-super-engine/output/nell995/tfidvectorizer

mkdir -p /home/barath/kg-super-engine/kg-super-engine/output/nell995/sentence

# Build dataset for nell995
# Cluster with TFID Vectorizer
python3 relationship-visualizer/data_preprocessing.py --data_path /home/barath/kg-super-engine/kg-super-engine/nell995 --data nell995 --output_dir /home/barath/kg-super-engine/kg-super-engine/output/nell995/tfidvectorizer
# Cluster with Sentence Embedding
python3 relationship-visualizer/data_preprocessing.py --cluster_type sentenceembedding --data_path /home/barath/kg-super-engine/kg-super-engine/nell995 --data nell995 --output_dir /home/barath/kg-super-engine/kg-super-engine/output/nell995/sentence
