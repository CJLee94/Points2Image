"""
Generate synthetic image/seg pairs on the fly 
"""
import time
import numpy as np
import torch
from options.train_options import TrainOptions
from util.post_proc import process
from data import create_dataset
from models import create_model
import h5py
import os
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
from tqdm import tqdm


def center_crop(data_dict, crop_size=1000):
    cropper = transforms.CenterCrop(crop_size)
    data_dict['A'] = cropper(data_dict['A'])
    data_dict['B'] = cropper(data_dict['B'])
    return data_dict

def run_inference(model, data, input_size=1000, patch_size=256, overlap=8):
    if patch_size == input_size:
        model.set_input(data)  # unpack data from data loader
        model.test()           # run inference
        visuals = model.get_current_visuals()  # get image results
        return visuals
    else:
        gen_patches = list()
        real_patches = list()

        for i in np.arange(0, input_size, overlap):
            for j in np.arange(0, input_size, patch_size-overlap):
                if i + patch_size > input_size or j + patch_size > input_size:
                    continue
                patch_data = dict()
                for key in data.keys():
                    if len(data[key].shape) == 4:
                        assert data[key].shape[2] == data[key].shape[3], 'only support w=h for now.'
                        patch_data[key] = data[key][:, :, i:i+patch_size, j:j+patch_size]
                        #print('patch', i, i+patch_size, j, j+patch_size, key, patch_data[key].shape)  
                    else:
                        patch_data[key] = data[key]
                model.set_input(patch_data)
                with torch.no_grad():
                    gen_patch = model.netG_A(model.real_A)  # G_A(A)
                    real_path = model.real_B
                gen_patch_numpy = gen_patch[0].detach().cpu().numpy()
                gen_patches.append(gen_patch_numpy[None, ...])
                real_patch_numpy = real_path[0].detach().cpu().numpy()
                real_patches.append(real_patch_numpy[None, ...])
                
        gen_patches = np.concatenate(gen_patches, axis=0)
        real_patches = np.concatenate(real_patches, axis=0)
        print(gen_patches.shape, real_patches.shape)
        return gen_patches, real_patches

"""
ckpt_dir="/scratch/mr5295/projects/Points2Image/CycleGAN/checkpoints_0227/"
data_root="/scratch/mr5295/data/point2image/processed_data/MoNuSeg_train_v4_enhanced_pcorrected.h5"

ckpt_dir="/home/mengwei/redwood_research/Points2Image/CycleGAN/checkpoints_0227/"
data_root="/home/mengwei/redwood_research/processed_data/MoNuSeg_train_v4_enhanced_pcorrected.h5"

ckpt_dir="/scratch/mr5295/projects/Points2Image/CycleGAN/checkpoints/"
data_root="/scratch/mr5295/data/point2image/processed_data/MoNuSeg_train_v3.h5"

name="Ga_resnet_Gb_resnet_bs12_colormask_v3"
name="v4_align_Ga_oasis_noise1_Gb_oasis_noise1_cyc5"
name="v4_align_Ga_oasis_synth_Gb_oasis_synth_cyc5"
name="v4_align_Ga_resnet_Gb_resnet_cyclegan_w_segloss_w_ploss"
name="v4_align_Ga_resnet_Gb_resnet_cyclegan_w_segloss"
name="v4_align_Ga_oasis_synth_Gb_hovernet_cyc5"
name="v4_align_Ga_resnet_Gb_resnet_cyclegan_w_segloss"

python generate_datasets_for_inception.py \
--train_opt_file ${ckpt_dir}/${name}/train_opt.txt \
--dataroot ${data_root}

cd util/gan-metrics-pytorch/
python kid_score.py \
--true ${ckpt_dir}/${name}/generated_patches/real.npy \
--fake ${ckpt_dir}/${name}/generated_patches/generated.npy \
--batch-size 32 \
--gpu 0

python fid_score.py \
--true ${ckpt_dir}/${name}/generated_patches/real.npy \
--fake ${ckpt_dir}/${name}/generated_patches/generated.npy \
--batch-size 32 \
--gpu 0

"""
if __name__ == '__main__':
    opt = TrainOptions().load_opt()   # get training options
    opt.batch_size = 1
    opt.crop_size = 1000
    opt.no_flip = True
    opt.serial_batches = True
    opt.preprocess = ''
    opt.checkpoints_dir = os.path.split(os.path.split(opt.train_opt_file)[0])[0]
    dataset = create_dataset(opt)  # create a dataset given opt.dataset_mode and other options
    #print(dataset.dataset.dir)
    #with h5py.File(os.path.join(dataset.dataset.dir), 'r') as h5f_r:
    #    uncropped_fake_masks = torch.from_numpy(h5f_r['gen_instance_masks'][...,0].astype(np.int64))

    opt.crop_size = 256
    model = create_model(opt)      # create a cyclegan model
    model.isTrain = False
    model.setup(opt)               # regular setup: load and print networks; create schedulers
    model.eval()

    save_subdir = os.path.join(model.save_dir, 'generated_patches')
    os.makedirs(save_subdir, exist_ok=True)
    
    fake_images_all = list()
    real_images_all = list()
    for i, data in tqdm(enumerate(dataset)):  # inner loop within one epoch
        gen_patches, real_patches = run_inference(model, data, 1000, 256, 24)
        '''sanity check'''
        fig, axes = plt.subplots(2,2,figsize=(10,10))
        axes[0, 0].imshow(np.transpose(0.5*(1+gen_patches[0]), (1,2,0)))
        axes[0, 1].imshow(np.transpose(0.5*(1+real_patches[0]), (1,2,0)))
        axes[1, 0].imshow(np.transpose(0.5*(1+gen_patches[1]), (1,2,0)))
        axes[1, 1].imshow(np.transpose(0.5*(1+real_patches[1]), (1,2,0)))

        fig.savefig(os.path.join(save_subdir,'sample_%d.jpg'%(i)), dpi=200)
        plt.close()

        assert(gen_patches.shape == real_patches.shape)
        fake_images_all.append(gen_patches)
        real_images_all.append(real_patches)

    fake_images_all = np.concatenate(fake_images_all, axis=0)
    real_images_all = np.concatenate(real_images_all, axis=0)

    print(fake_images_all.shape, real_images_all.shape)
    np.save(os.path.join(save_subdir, 'generated'), fake_images_all)
    np.save(os.path.join(save_subdir, 'real'), real_images_all)

