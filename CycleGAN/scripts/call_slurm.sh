#!/bin/bash
export ckpt_dir="../checkpoints/"
export name="basic_netD_oasis_netGa_resnet_9blocks_netGb_cyclegan"


mkdir -v ${ckpt_dir}/${name}/
sbatch --job-name=${name} \
--output=${ckpt_dir}/${name}/train.out \
--error=${ckpt_dir}/${name}/train.error \
--export=ALL,ckpt_dir=${ckpt_dir},name=${name} slurm_train.sh

echo "submitted ${name}"
