#!/bin/bash

#SBATCH --partition=gpu
#SBATCH --gpus-per-node=1
#SBATCH --job-name=Trainmetatitan
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=3
#SBATCH --time=1:30:00
#SBATCH --mem=32000M
#SBATCH --output=wikidata5mprep.out

module purge
module load 2021
module load Anaconda3/2021.05

# /home/barath/blogwriter/AutoBlogWriter
cd $HOME/kg-super-engine/kg-super-engine
eval "$(conda shell.bash hook)"
conda activate kg-super

mkdir -p  $HOME/kg-super-engine/kg-super-engine/output/wikidata5m/tfidvectorizer
mkdir -p  $HOME/kg-super-engine/kg-super-engine/output/wikidata5m/sentence

# # Cluster with TFID Vectorizer
# python3 relationship-visualizer/data_preprocessing.py
# # Cluster with Sentence Embedding
# python3 relationship-visualizer/data_preprocessing.py --cluster_type sentenceembedding --output_dir $HOME/kg-super-engine/kg-super-engine/output/fb15k237/sentence

# Build dataset for wikidata5m
# Cluster with TFID Vectorizer
python3 relationship-visualizer/data_preprocessing.py --data_path $HOME/kg-super-engine/kg-super-engine/wikidata5m --data wikidata5m --output_dir $HOME/kg-super-engine/kg-super-engine/output/wikidata5m/tfidvectorizer
# Cluster with Sentence Embedding
python3 relationship-visualizer/data_preprocessing.py --cluster_type sentenceembedding --data_path $HOME/kg-super-engine/kg-super-engine/wikidata5m --data wikidata5m --output_dir $HOME/kg-super-engine/kg-super-engine/output/wikidata5m/sentence