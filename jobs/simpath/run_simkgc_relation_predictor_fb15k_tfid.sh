#!/bin/bash

#SBATCH --partition=gpu
#SBATCH --gpus-per-node=1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=18
#SBATCH --job-name=SIMKGCRelfb15ktfid
#SBATCH --time=03:00:00
#SBATCH --mem=80G
#SBATCH --output=simkgcrelationtfid.out

module purge
module load 2021
module load Anaconda3/2021.05

# /home/barath/blogwriter/AutoBlogWriter
cd $HOME/kg-super-engine/kg-super-engine/simpath/main
eval "$(conda shell.bash hook)"
conda activate kg-rmpi2


# Copy the files to a local folder that can be used later
cp -r /home/barath/kg-super-engine/kg-super-engine/simpath/main/data/FB15k237/* /home/barath/kg-super-engine/kg-super-engine/simpath/main/data/FB15k237_tfid_radial/

cp -r /home/barath/kg-super-engine/kg-super-engine/output/fb15k237/radial_cluster/random_centroid_1/*.txt /home/barath/kg-super-engine/kg-super-engine/simpath/main/data/FB15k237_tfid_radial/


python3 -u preprocess.py \
--task FB15k237 \
--train-path "$HOME/kg-super-engine/kg-super-engine/simpath/main/data/FB15k237_tfid_radial/train.txt" \
--valid-path "$HOME/kg-super-engine/kg-super-engine/simpath/main/data/FB15k237_tfid_radial/valid.txt" \
--test-path "$HOME/kg-super-engine/kg-super-engine/simpath/main/data/FB15k237_tfid_radial/test.txt"


# Not adding a negative sampling until I figure out what it does
python3 -u main.py --model-dir "$HOME/kg-super-engine/kg-super-engine/simpath/main/tfid_radial" --pretrained-model bert-base-uncased --pooling mean --lr 1e-5 --use-link-graph --train-path "$HOME/kg-super-engine/kg-super-engine/simpath/main/data/FB15k237_tfid_radial/train.txt.json" --valid-path "$HOME/kg-super-engine/kg-super-engine/simpath/main/data/FB15k237_tfid_radial/valid.txt.json" --task FB15k237 --batch-size 512 --print-freq 20 --additive-margin 0.02 --use-amp  --finetune-t --pre-batch 2 --epochs 20 --workers 4 --max-to-keep 5

# Make sure to include the evaluation also as part of this
python3 -u evaluate.py --task FB15K237  --is-test  --eval-model-path "$HOME/kg-super-engine/kg-super-engine/simpath/main/tfid_radial/model_best.mdl"  --neighbor-weight 0.05  --rerank-n-hop 2 --train-path "$HOME/kg-super-engine/kg-super-engine/simpath/main/data/FB15k237_tfid_radial/train.txt.json" --valid-path "$HOME/kg-super-engine/kg-super-engine/simpath/main/data/FB15k237_tfid_radial/valid.txt.json"