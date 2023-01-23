#!/bin/bash

#SBATCH --partition=gpu
#SBATCH --gpus-per-node=1
#SBATCH --job-name=KGLinkBaseline
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=3
#SBATCH --time=11:59:59
#SBATCH --mem=128000M
#SBATCH --output=kgbertlinkpredbaseline.out

module purge
module load 2021
module load Anaconda3/2021.05

# /home/barath/blogwriter/AutoBlogWriter
cd $HOME/kg-super-engine/kg-super-engine
eval "$(conda shell.bash hook)"
conda activate kg-super

python3 /home/barath/kg-super-engine/kg-bert/kg-bert/run_bert_link_prediction.py --task_name kg --do_train --do_eval --do_predict  --data_dir /home/barath/kg-super-engine/kg-bert-random-centroid_0 --bert_model bert-base-cased --max_seq_length 150 --train_batch_size 32  --learning_rate 5e-5  --num_train_epochs 5.0  --output_dir /home/barath/kg-super-engine/kg-bert/kg-bert/linkpred_random_centroid_0/   --gradient_accumulation_steps 1  --eval_batch_size 1500