#!/bin/bash

#SBATCH --partition=gpu
#SBATCH --gpus-per-node=1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=18
#SBATCH --job-name=RMPILinkPredTrain
#SBATCH --time=12:00:00
#SBATCH --mem=60G
#SBATCH --output=rmpilinkradial2train.out

module purge
module load 2021
module load Anaconda3/2021.05

# /home/barath/blogwriter/AutoBlogWriter
cd $HOME/kg-super-engine/kg-super-engine/RMPI
eval "$(conda shell.bash hook)"
conda activate kg-rmpi2

mkdir -p $HOME/kg-super-engine/kg-super-engine/RMPI/data/nell_radial_centroid_2/

# Copy the files to a local folder that can be used later
cp -r $HOME/kg-super-engine/kg-super-engine/RMPI/data/nell_v3/* $HOME/kg-super-engine/kg-super-engine/RMPI/data/nell_radial_centroid_2/

cp -r $HOME/kg-super-engine/kg-super-engine/output/nell995/radial_cluster/random_centroid_2/*.txt $HOME/kg-super-engine/kg-super-engine/RMPI/data/nell_radial_centroid_2/

echo "TRIGGERING THE TRAINING FOR RADIAL CLUSTER CENTROID 1 \n"

python RMPI/train.py -d nell_radial_centroid_2 -e nell_v2_RMPI_centroid1 --ablation 0

python RMPI/test_ranking_F.py -d nell_radial_centroid_2 -e nell_v2_RMPI_centroid1 --ablation 0
