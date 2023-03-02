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

def run_inference(model, data, input_size=1000, patch_size=256, overlap=8, sample_times=1, max_image=10000):
    if patch_size == input_size:
        model.set_input(data)  # unpack data from data loader
        model.test()           # run inference
        visuals = model.get_current_visuals()  # get image results
        return visuals
    else:
        gen_patches = list()
        real_patches = list()
        std_patches = list()
        mean_patches = list()
        input_patches = list()
        synthseg_patches = list()

        for i in np.arange(0, input_size, overlap):
            for j in np.arange(0, input_size, patch_size-overlap):
                if len(real_patches) >= max_image or i + patch_size > input_size or j + patch_size > input_size:
                    continue
                # construct a data dict for the model
                patch_data = dict()
                for key in data.keys():
                    if len(data[key].shape) == 4:
                        assert data[key].shape[2] == data[key].shape[3], 'only support w=h for now.'
                        patch_data[key] = data[key][:, :, i:i+patch_size, j:j+patch_size]
                        #print('patch', i, i+patch_size, j, j+patch_size, key, patch_data[key].shape)  
                    else:
                        patch_data[key] = data[key]
                # sample multiple times 
                sample_patches = list()
                sample_synthseg_patches = list()
                for sample_iter in range(sample_times):
                    model.set_input(patch_data)
                    with torch.no_grad():
                        gen_patch = model.netG_A(model.real_A)  # G_A(A)
                        real_patch = model.real_B
                        input_patch = model.real_A
                        if model.opt.use_synthseg:
                            synthseg_patch = model.netG_A.module.z_synthseg
                            synthseg_patch_numpy = synthseg_patch[0].detach().cpu().numpy()
                    gen_patch_numpy = gen_patch[0].detach().cpu().numpy()
                    sample_patches.append(gen_patch_numpy[None, ...])
                    if model.opt.use_synthseg:
                        sample_synthseg_patches.append(synthseg_patch_numpy[None, ...])
                
                sample_patches = np.concatenate(sample_patches, axis=0)
                if model.opt.use_synthseg:
                    sample_synthseg_patches = np.concatenate(sample_synthseg_patches, axis=0)
                # get the std of the multiple samples 
                std_image = np.std(sample_patches, axis=0, keepdims=True)
                mean_image = np.mean(sample_patches, axis=0, keepdims=True)

                std_patches.append(std_image)
                mean_patches.append(mean_image)
                
                gen_patches.append(sample_patches[None, ...])
                if model.opt.use_synthseg:
                    synthseg_patches.append(sample_synthseg_patches[None, ...])

                real_patch_numpy = real_patch[0].detach().cpu().numpy()
                real_patches.append(real_patch_numpy[None, ...])
                input_patch_numpy = input_patch[0].detach().cpu().numpy()
                input_patches.append(input_patch_numpy[None, ...])


        gen_patches = np.concatenate(gen_patches, axis=0)
        if model.opt.use_synthseg:
            synthseg_patches = np.concatenate(synthseg_patches, axis=0)
        real_patches = np.concatenate(real_patches, axis=0)
        input_patches = np.concatenate(input_patches, axis=0)

        std_patches = np.concatenate(std_patches, axis=0)
        mean_patches = np.concatenate(mean_patches, axis=0)
        print(gen_patches.shape, real_patches.shape, std_patches.shape, mean_patches.shape)
        #(max_image, sample_times, 3, 256, 256) (max_image, 3, 256, 256) (max_image, 3, 256, 256)
        return input_patches, gen_patches, real_patches, std_patches, mean_patches, synthseg_patches

"""
ckpt_dir="/scratch/mr5295/projects/Points2Image/CycleGAN/checkpoints_0227/"
data_root="/scratch/mr5295/data/point2image/processed_data/MoNuSeg_train_v4_enhanced_pcorrected.h5"
ckpt_dir="/home/mengwei/redwood_research/Points2Image/CycleGAN/checkpoints_0227/"
data_root="/home/mengwei/redwood_research/processed_data/MoNuSeg_train_v4_enhanced_pcorrected.h5"

name="v4_unalign_Ga_resnet_Gb_resnet_cyclegan_w_segloss"
name="v4_align_Ga_oasis_noise1_Gb_oasis_noise1_cyc5"
name="v4_align_Ga_oasis_synth_Gb_oasis_synth_cyc5"
name="v4_align_Ga_oasis_synth_Gb_hovernet_cyc5"
name="v4_align_Ga_oasis_noise1_Gb_hover_cyc5"
python generate_datasets_for_diversity.py \
--train_opt_file ${ckpt_dir}/${name}/train_opt.txt \
--dataroot ${data_root}


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
    with h5py.File(os.path.join(dataset.dataset.dir), 'r') as h5f_r:
        uncropped_fake_masks = torch.from_numpy(h5f_r['gen_instance_masks'][...,0].astype(np.int64))

    opt.crop_size = 256
    model = create_model(opt)      # create a cyclegan model
    model.isTrain = False
    model.setup(opt)               # regular setup: load and print networks; create schedulers
    model.eval()

    save_subdir = os.path.join(model.save_dir, 'generated_multiple_patches')
    os.makedirs(save_subdir, exist_ok=True)
    
    fake_images_all = list()
    real_images_all = list()
    for i, data in tqdm(enumerate(dataset)):  # inner loop within one epoch
        input_patches, gen_patches, \
        real_patches, std_patches, \
        mean_patches, synthseg_patches = run_inference(model, 
                                                       data, 
                                                       1000, 256, 24, 
                                                       sample_times=10, max_image=2)
        '''sanity check'''
        if opt.use_synthseg:
            fig, axes = plt.subplots(3,4,figsize=(10,10))
        else:
            fig, axes = plt.subplots(2,4,figsize=(10,10))
        axes[0, 0].imshow(np.transpose(0.5*(1+gen_patches[0,0]), (1,2,0))), axes[0, 0].set_title('sample1')
        axes[0, 1].imshow(np.transpose(0.5*(1+gen_patches[0,1]), (1,2,0))), axes[0, 1].set_title('sample2')
        axes[0, 2].imshow(np.transpose(0.5*(1+gen_patches[0,2]), (1,2,0))), axes[0, 2].set_title('sample3')
        axes[0, 3].imshow(np.transpose(0.5*(1+gen_patches[0,3]), (1,2,0))), axes[0, 3].set_title('sample4')

        axes[1, 0].imshow(np.transpose(0.5*(1+input_patches[0]), (1,2,0))), axes[1, 0].set_title('input_mask')
        axes[1, 1].imshow(np.transpose(0.5*(1+real_patches[0]), (1,2,0))), axes[1, 1].set_title('real')
        axes[1, 2].imshow(np.transpose(0.5*(1+mean_patches[0]), (1,2,0))), axes[1, 2].set_title('mean')
        axes[1, 3].imshow(np.transpose(0.5*(1+std_patches[0]), (1,2,0))), axes[1, 3].set_title('std') #, plt.colorbar()
        
        if opt.use_synthseg:
            axes[2, 0].imshow(0.5*(1+synthseg_patches[0,0,0])), axes[2, 0].set_title('synthseg1')
            axes[2, 1].imshow(0.5*(1+synthseg_patches[0,1,0])), axes[2, 1].set_title('synthseg2')
            axes[2, 2].imshow(0.5*(1+synthseg_patches[0,2,0])), axes[2, 2].set_title('synthseg3')
            axes[2, 3].imshow(0.5*(1+synthseg_patches[0,3,0])), axes[2, 3].set_title('synthseg4')


        fig.savefig(os.path.join(save_subdir,'sample_%d.jpg'%(i)), dpi=200)
        plt.close()

        assert(std_patches.shape == real_patches.shape)
        fake_images_all.append(gen_patches)
        real_images_all.append(real_patches)

    fake_images_all = np.concatenate(fake_images_all, axis=0)
    real_images_all = np.concatenate(real_images_all, axis=0)

    print(fake_images_all.shape, real_images_all.shape)
    np.save(os.path.join(save_subdir, 'generated'), fake_images_all)
    np.save(os.path.join(save_subdir, 'real'), real_images_all)

