#!/bin/bash

#SBATCH --partition=gpu
#SBATCH --gpus-per-node=1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=18
#SBATCH --job-name=RMPILinkPredTrain
#SBATCH --time=06:00:00
#SBATCH --mem=60G
#SBATCH --output=rmpilinktrain.out

module purge
module load 2021
module load Anaconda3/2021.05

# /home/barath/blogwriter/AutoBlogWriter
cd $HOME/kg-super-engine/kg-super-engine/RMPI
eval "$(conda shell.bash hook)"
conda activate kg-rmpi2

# python TACT/train.py -d nell_v2 -e nell_v2_TACT_base --ablation 3
# python RMPI/train.py -d nell_v2 -e nell_v2_RMPI_base --ablation 0

python RMPI/test_ranking_F.py -d nell_v2_ind_v3_semi -e nell_v2_RMPI_base --ablation 0

python RMPI/test_ranking_P.py -d nell_v2_ind -e nell_v2_RMPI_base --ablation 0