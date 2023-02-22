#!/bin/bash
#SBATCH --time=20:00:00
#SBATCH --gres=gpu:v100:1
#SBATCH --mem=16G
#SBATCH --mail-type=ALL                    # Mail events (NONE, BEGIN, END, FAIL, ALL)
#SBATCH --mail-user=mengwei.ren@nyu.edu   # Where to send mail
#SBATCH -c 4

cp -v /scratch/mr5295/singularity/overlay-25GB-500K.ext3 /scratch/mr5295/singularity/overlay_cyclegan_${name}.ext3
singularity exec --nv --overlay /scratch/mr5295/singularity/overlay_cyclegan_${name}.ext3 /scratch/work/public/singularity/cuda11.1-cudnn8-devel-ubuntu18.04.sif /bin/bash -c "
source /ext3/miniconda3/etc/profile.d/conda.sh
export PATH=/ext3/miniconda3/bin:$PATH
conda activate pytorch-CycleGAN-and-pix2pix

CUDA_VISIBLE_DEVICES=0 python ../train.py \
--dataroot /scratch/mr5295/data/point2image/processed_data/MoNuSeg_train_v4_enhanced.h5 \
--dataset_mode instancemask \
--checkpoints_dir ${ckpt_dir} \
--name ${name} \
--batch_size 12 \
--norm instance \
--netG_A resnet_9blocks --netG_B unet_256 --netD basic_sn \
--ngf 64 \
--model instancecyclegan \
--pool_size 50 --no_dropout --phase train --preprocess crop \
--crop_size 256 --lambda_identity 0 --n_epochs 5000 \
--n_epochs_decay 1000 --input_nc 3 --output_nc 3 \
--save_latest_freq 100 --save_epoch_freq 500 \
--continue_train --epoch_count 4500
"

rm -v /scratch/mr5295/singularity/overlay_cyclegan_${name}.ext3