#!/bin/bash

#SBATCH --partition=gpu
#SBATCH --gpus-per-node=2
#SBATCH --nodes=1
#SBATCH --job-name=KGLinkPredEval
#SBATCH --cpus-per-task=3
#SBATCH --time=23:00:00
#SBATCH --output=kglinkpredeval.out

module purge
module load 2021
module load Anaconda3/2021.05

# /home/barath/blogwriter/AutoBlogWriter
cd $HOME/kg-super-engine/kg-super-engine
eval "$(conda shell.bash hook)"
conda activate kg-super

python3 $HOME/kg-super-engine/kg-bert/kg-bert/run_bert_link_prediction.py --task_name kg --do_eval --do_predict  --data_dir $HOME/kg-super-engine/kg-bert-random-centroid_1 --bert_model bert-base-cased --max_seq_length 150 --train_batch_size 32  --learning_rate 5e-5  --num_train_epochs 5.0  --output_dir $HOME/kg-super-engine/kg-bert/kg-bert/linkpred_random_centroid_1/   --gradient_accumulation_steps 1  --eval_batch_size 1500