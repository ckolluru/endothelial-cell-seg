#!/bin/bash
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --time=10:00:00
#SBATCH --mem=100g
#SBATCH -p gpuk40 --gres=gpu:2
#SBATCH --job-name=u_net_cell_segmentation
#SBATCH --mail-user=cxk340@case.edu
#SBATCH --mail-type=begin


module load tensorflow

singularity exec $TENSORFLOW python /home/cxk340/cell_seg_software/unet.py

