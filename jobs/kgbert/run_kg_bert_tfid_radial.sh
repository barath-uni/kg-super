#!/bin/bash

#SBATCH --partition=gpu
#SBATCH --gpus-per-node=1
#SBATCH --job-name=KGRandomCentroid
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=3
#SBATCH --time=47:59:59
#SBATCH --mem=128000M
#SBATCH --output=kgberttrainrandomcentroid.out

module purge
module load 2021
module load Anaconda3/2021.05

# /home/barath/blogwriter/AutoBlogWriter
cd $HOME/kg-super-engine/kg-super-engine
eval "$(conda shell.bash hook)"
conda activate kg-super


# Copy the files to a local folder that can be used later
cp -r $HOME/kg-super-engine/kg-bert/kg-bert/data/FB15k-237/* $HOME/kg-super-engine/kg-bert-random-centroid/

cp -r $HOME/kg-super-engine/kg-super-engine/output/fb15k237/radial_cluster/random_centroid_1/*.tsv $HOME/kg-super-engine/kg-bert-random-centroid/

# Trigger the relation prediction from the parent folder kg-bert

python3 $HOME/kg-super-engine/kg-bert/kg-bert/run_bert_relation_prediction.py  --task_name kg --do_train --do_eval --do_predict --data_dir $HOME/kg-super-engine/kg-bert-random-centroid --bert_model bert-base-cased --max_seq_length 25 --train_batch_size 32  --num_train_epochs 20.0  --output_dir $HOME/kg-super-engine/kg-bert/kg-bert/output_fb15k23_randomcentroid/   --gradient_accumulation_steps 1 --eval_batch_size 512