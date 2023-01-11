#!/bin/bash

#SBATCH --partition=gpu
#SBATCH --gpus-per-node=2
#SBATCH --nodes=1
#SBATCH --job-name=KGBertSentenceRadial
#SBATCH --cpus-per-task=3
#SBATCH --time=1:00:00
#SBATCH --output=kgbertqual.out

module purge
module load 2021
module load Anaconda3/2021.05

# /home/barath/blogwriter/AutoBlogWriter
cd $HOME/kg-super-engine/kg-super-engine
eval "$(conda shell.bash hook)"
conda activate kg-super

# Trigger the relation prediction from the parent folder kg-bert

python3 /home/barath/kg-super-engine/kg-bert/kg-bert/run_bert_relation_prediction.py  --task_name kg --do_eval --do_predict --data_dir /home/barath/kg-super-engine/kg-bert-sentece-random-centroid_0 --bert_model bert-base-cased --max_seq_length 25 --train_batch_size 32  --num_train_epochs 20.0  --output_dir /home/barath/kg-super-engine/kg-bert/kg-bert/output_fb15k23_randomcentroidsentence0/   --gradient_accumulation_steps 1 --eval_batch_size 512