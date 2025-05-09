#!/bin/bash
#SBATCH --job-name=mcqa_classifier
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --partition=gengpu
#SBATCH --account=p32368 # Removed for home allocation

#SBATCH --output=logs/mcqa_%j.out
#SBATCH --error=logs/mcqa_%j.err

#SBATCH --mail-user=vrishanishah2025@u.northwestern.edu
#SBATCH --mail-type=BEGIN,END,FAIL

# === Load Anaconda and activate conda ===
module purge
module load anaconda3

# Force use of your conda installation
export PATH=/home/jci0365/miniconda3/bin:$PATH
source /home/jci0365/miniconda3/etc/profile.d/conda.sh
conda activate vae

# === Confirm environment ===
echo "Running on host: $(hostname)"
echo "Python in use: $(which python)"
python -c "import torch; print('CUDA available:', torch.cuda.is_available())"

# === Run the script ===
python training_classifier.py
