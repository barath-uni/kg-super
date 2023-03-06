#SBATCH --cpus-per-task=18
#SBATCH --job-name=SIMKGCPreprocess
#SBATCH --time=01:00:00
#SBATCH --mem=60G
#SBATCH --output=simkgcpreprocess.out

module purge
module load 2021
module load Anaconda3/2021.05

# /home/barath/blogwriter/AutoBlogWriter
cd $HOME/kg-super-engine/kg-super-engine/simkgc/SimKGC
eval "$(conda shell.bash hook)"
conda activate kg-rmpi2

python3 -u preprocess.py \
--task FB15k237 \
--train-path "$HOME/kg-super-engine/kg-super-engine/simkgc/SimKGC/data/FB15k237/train.txt" \
--valid-path "$HOME/kg-super-engine/kg-super-engine/simkgc/SimKGC/data/FB15k237/valid.txt" \
--test-path "$HOME/kg-super-engine/kg-super-engine/simkgc/SimKGC/data/FB15k237/test.txt"
