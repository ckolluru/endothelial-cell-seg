#!/bin/bash
#SBATCH -N 1
#SBATCH --cpus-per-task 12
#SBATCH --time=72:00:00
#SBATCH --mem=186g
#SBATCH -p gpu -C gpup100 --gres=gpu:2
#SBATCH --job-name=unet_ec_segment
#SBATCH --mail-user=cxk340@case.edu
#SBATCH --mail-type=end

#Use pre-existing HPC modules
export OMP_NUM_THREADS=1
module swap intel gcc
module load tensorflow
python /home/cxk340/endothelial-cell-seg/unet.py

#Use singularity image method
#module load gcc cuda singularity
#singularity exec --nv /home/cxk340/OCT_DL/singularity_image/keras_tf.img python /home/cxk340/endothelial-cell-seg/unet.py
