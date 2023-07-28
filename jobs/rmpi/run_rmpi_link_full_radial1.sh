#!/bin/bash

#SBATCH --partition=gpu
#SBATCH --gpus-per-node=1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=18
#SBATCH --job-name=RMPILinkPredTrain
#SBATCH --time=12:00:00
#SBATCH --mem=60G
#SBATCH --output=rmpilinkradial1train.out

module purge
module load 2021
module load Anaconda3/2021.05

cd $HOME/kg-super-engine/kg-super-engine/RMPI
eval "$(conda shell.bash hook)"
conda activate kg-rmpi2

mkdir -p $HOME/kg-super-engine/kg-super-engine/RMPI/data/nell_radial_centroid_1/

# Copy the files to a local folder that can be used later
cp -r $HOME/kg-super-engine/kg-super-engine/RMPI/data/nell_v3/* $HOME/kg-super-engine/kg-super-engine/RMPI/data/nell_radial_centroid_1/

cp -r $HOME/kg-super-engine/kg-super-engine/output/nell995/radial_cluster/random_centroid_1/*.txt $HOME/kg-super-engine/kg-super-engine/RMPI/data/nell_radial_centroid_1/

echo "TRIGGERING THE TRAINING FOR RADIAL CLUSTER CENTROID 1 \n"

python RMPI/train.py -d nell_radial_centroid_1 -e nell_v2_RMPI_centroid1 --ablation 0

python RMPI/test_ranking_F.py -d nell_radial_centroid_1 -e nell_v2_RMPI_centroid1 --ablation 0
