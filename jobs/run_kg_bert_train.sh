#!/bin/bash

#SBATCH --partition=gpu_titanrtx_shared
#SBATCH --gres=gpu:1
#SBATCH --job-name=KGBertSentence
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=3
#SBATCH --time=12:00:00
#SBATCH --mem=32000M
#SBATCH --output=kgberttrain.out

module purge
module load 2021
module load Anaconda3/2021.05

# /home/barath/blogwriter/AutoBlogWriter
cd $HOME/kg-super-engine/kg-super-engine
eval "$(conda shell.bash hook)"
conda activate kg-super

cp -r /home/barath/kg-super-engine/kg-super-engine/output/wikidata5m/sentence/*.tsv /home/barath/kg-super-engine/kg-bert/kg-bert/data/FB15k-237/

# Trigger the relation prediction from the parent folder kg-bert

python3 /home/barath/kg-super-engine/kg-bert/kg-bert/run_bert_relation_prediction.py 
--task_name kg-bert-sentence
--do_train
--do_eval
--do_predict
--data_dir /home/barath/kg-super-engine/kg-bert/kg-bert/FB15k-237
--bert_model bert-base-cased
--max_seq_length 25
--train_batch_size 32 
--learning_rate 5e-5 
--num_train_epochs 20.0 
--output_dir /home/barath/kg-super-engine/kg-bert/kg-bert/output_fb15k237/  
--gradient_accumulation_steps 1
--eval_batch_size 512