#!/bin/bash
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --time=10:00:00
#SBATCH --mem=100g
#SBATCH -p gpuk40 --gres=gpu:2
#SBATCH --job-name=u_net_cell_segmentation
#SBATCH --mail-user=cxk340@case.edu
#SBATCH --mail-type=begin

#Use pre-existing HPC modules
export OMP_NUM_THREADS=1
module swap intel gcc
module load tensorflow

python /home/cxk340/endothelial-cell-seg/unet.py

#Use singularity image method
module load gcc cuda singularity
singularity shell --nv /home/cxk340/OCT_DL/singularity_image/keras_tf.img
python /home/cxk340/endothelial-cell-seg/unet.py