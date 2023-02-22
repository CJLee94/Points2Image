#!/bin/bash
export ckpt_dir="../checkpoints/"
export name="Ga_resnet_Gb_unet_D_basic_sn_bs12_ngf64_v4"


mkdir -v ${ckpt_dir}/${name}/
cp -v ./slurm_train.sh ${ckpt_dir}/${name}/

sbatch --job-name=${name} \
--output=${ckpt_dir}/${name}/train.out \
--error=${ckpt_dir}/${name}/train.error \
--export=ALL,ckpt_dir=${ckpt_dir},name=${name} slurm_train.sh

echo "submitted ${name}"
