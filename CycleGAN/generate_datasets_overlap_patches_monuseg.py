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
import os,sys
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
from tqdm import tqdm
from util.visualizer import create_group_fig
from util.targets import gen_targets
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
np.random.seed(2023)

def center_crop(data_dict, crop_size=1000):
    cropper = transforms.CenterCrop(crop_size)
    data_dict['A'] = cropper(data_dict['A'])
    data_dict['B'] = cropper(data_dict['B'])
    return data_dict


def get_item_synth_mask(hd, opt, image_index, mask_index):
    """Return a data point and its metadata information.

    Parameters:
        index - - a random integer for data indexing

    Returns a dictionary that contains A, B, A_paths and B_paths
        A (tensor) - - the source image
        B (tensor) - - the target image
        A_paths (str) - - image paths
        B_paths (str) - - image paths (same as A_paths)
    """
    
    #hd_len = hd['images'].shape[0]
    # get the instance mask (scale 0 - N)
    A_img = hd['gen_instance_masks'][image_index][..., None]  # W x H x 1
    A_img = np.transpose(A_img, (2,0,1)).astype(np.float32)
    A_img = torch.from_numpy(A_img)
    
    b_index = image_index
    B_img = hd['images'][b_index]
    B_img = 2 * B_img.astype(np.float32) / 255. - 1.0
    B_img = np.transpose(B_img, (2,0,1))
    B_img = torch.from_numpy(B_img)

    P_img = hd['point_masks'][image_index]
    P_img = P_img.astype(np.float32) / 255.
    P_img = torch.from_numpy(P_img[None, ...])

    # get hv map and binary mask
    A_instance = A_img[0].numpy().astype(np.int32)
    A_masks = gen_targets(A_instance, crop_shape=A_instance.shape)
    if opt.norm_seg:
        A_masks['np_map'] = 2 * A_masks['np_map'] - 1.
    #print(A_masks['hv_map'].shape, A_masks['np_map'].shape)  # crop x crop x 2,  crop x crop
    A = np.concatenate((A_masks['hv_map'], A_masks['np_map'][..., None]), axis=-1)
    A = np.transpose(A, (2,0,1))
    A = torch.from_numpy(A).float()
    return_dict = {'A_instance': A_img.unsqueeze(0), 
                   'A': A.unsqueeze(0), 
                   'B': B_img.unsqueeze(0), 
                   'P': P_img.unsqueeze(0),
                   'A_paths': image_index,
                   'B_paths': b_index}
    return return_dict


def get_item_single_mask(hd, opt, image_index, mask_index):
    """Return a data point and its metadata information.

    Parameters:
        index - - a random integer for data indexing

    Returns a dictionary that contains A, B, A_paths and B_paths
        A (tensor) - - the source image
        B (tensor) - - the target image
        A_paths (str) - - image paths
        B_paths (str) - - image paths (same as A_paths)
    """
    
    hd_len = hd['images'].shape[0]
    # get the instance mask (scale 0 - N)
    A_img = hd['gen_instance_masks'][image_index][..., 0:1]  # W x H x 1
    A_img = np.transpose(A_img, (2,0,1)).astype(np.float32)
    A_img = torch.from_numpy(A_img)
    
    P_img = hd['points_masks'][image_index]
    P_img = P_img.astype(np.float32) / 255.
    P_img = torch.from_numpy(P_img[None, ...])

    b_index = image_index
          
    B_img = hd['images'][b_index]
    B_img = 2 * B_img.astype(np.float32) / 255. - 1.0
    B_img = np.transpose(B_img, (2,0,1))
    B_img = torch.from_numpy(B_img)

    # get hv map and binary mask
    A_instance = A_img[0].numpy().astype(np.int32)
    A_masks = gen_targets(A_instance, crop_shape=A_instance.shape)
    if opt.norm_seg:
        A_masks['np_map'] = 2 * A_masks['np_map'] - 1.
    #print(A_masks['hv_map'].shape, A_masks['np_map'].shape)  # crop x crop x 2,  crop x crop
    A = np.concatenate((A_masks['hv_map'], A_masks['np_map'][..., None]), axis=-1)
    A = np.transpose(A, (2,0,1))
    A = torch.from_numpy(A).float()
    return_dict = {'A_instance': A_img.unsqueeze(0), 
                   'A': A.unsqueeze(0), 
                   'B': B_img.unsqueeze(0), 
                   'P': P_img.unsqueeze(0),
                   'A_paths': image_index,
                   'B_paths': b_index}
    return return_dict


def get_item(hd, opt, image_index, mask_index):
    """Return a data point and its metadata information.

    Parameters:
        index - - a random integer for data indexing

    Returns a dictionary that contains A, B, A_paths and B_paths
        A (tensor) - - the source image
        B (tensor) - - the target image
        A_paths (str) - - image paths
        B_paths (str) - - image paths (same as A_paths)
    """
    
    hd_len = hd['images'].shape[0]
    # get the instance mask (scale 0 - N)
    A_img = hd['gen_instance_masks'][image_index]  # number of masks x W x H
    number_of_masks_per_images = opt.num_masks #A_img.shape[0]
    assert number_of_masks_per_images <= A_img.shape[0]
    
    A_img = A_img[mask_index: mask_index+1].astype(np.float32)  # 1 x W x H
    A_img = torch.from_numpy(A_img)
    
    P_img = hd['points_masks'][image_index]
    P_img = P_img.astype(np.float32) / 255.
    P_img = torch.from_numpy(P_img[None, ...])

    b_index = image_index
    B_img = hd['images'][b_index]
    B_img = 2 * B_img.astype(np.float32) / 255. - 1.0
    B_img = np.transpose(B_img, (2,0,1))
    B_img = torch.from_numpy(B_img)
    B_instance = hd['instance_masks'][b_index]

    # get hv map and binary mask
    A_instance = A_img[0].numpy().astype(np.int32)
    A_masks = gen_targets(A_instance, crop_shape=A_instance.shape)
    if opt.norm_seg:
        A_masks['np_map'] = 2 * A_masks['np_map'] - 1.
    #print(A_masks['hv_map'].shape, A_masks['np_map'].shape)  # crop x crop x 2,  crop x crop
    A = np.concatenate((A_masks['hv_map'], A_masks['np_map'][..., None]), axis=-1)
    A = np.transpose(A, (2,0,1))
    A = torch.from_numpy(A).float()
    return_dict = {'A_instance': A_img.unsqueeze(0), 
                   'A': A.unsqueeze(0), 
                   'B': B_img.unsqueeze(0), 
                   'B_instance': torch.from_numpy(B_instance).unsqueeze(0),
                   'P': P_img.unsqueeze(0),
                   'A_paths': image_index,
                   'B_paths': b_index}
    return return_dict


def run_inference(model, data, 
                  input_size=1000, patch_size=256, overlap=8, sample_times=1, max_image=10000,
                  return_minimum=False):
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
        point_patches = list()
        instance_patches = list()
        real_instance_patches = list()

        for i in np.arange(0, input_size, overlap):
            for j in np.arange(0, input_size, overlap):
                if len(real_patches) >= max_image or (i + patch_size > input_size) or (j + patch_size > input_size):
                    continue
                #print(i,j)
                # construct a data dict for the model
                patch_data = dict()
                for key in data.keys():
                    if key in ['A_paths', 'B_paths']:
                        patch_data[key] = data[key]
                        continue
                    if len(data[key].shape) == 4:
                        assert data[key].shape[2] == data[key].shape[3], 'only support w=h for now.'
                        patch_data[key] = data[key][:, :, i:i+patch_size, j:j+patch_size]
                        #print('patch', i, i+patch_size, j, j+patch_size, key, patch_data[key].shape)  
                    elif len(data[key].shape) == 3:
                        patch_data[key] = data[key][:, i:i+patch_size, j:j+patch_size]
                    else:
                        patch_data[key] = data[key]
                    #print(key, patch_data[key].shape)
                # sample multiple times 
                patch_data['A'] = patch_data['A'].repeat(sample_times, 1, 1, 1)
                #for sample_iter in range(sample_times):
                model.set_input(patch_data)
                with torch.no_grad():
                    gen_patch = model.netG_A(model.real_A)  # G_A(A)
                    real_patch = model.real_B
                    input_patch = model.real_A
                    if 'P' in data.keys():
                        point_patch = patch_data['P'] #model.real_P

                sample_patches = gen_patch.detach().cpu().numpy()
                
                # get the std of the multiple samples 
                std_image = np.std(sample_patches, axis=0, keepdims=True)

                std_patches.append(std_image)
                gen_patches.append(sample_patches[None, ...])

                real_patch_numpy = real_patch[0].detach().cpu().numpy()
                real_patches.append(real_patch_numpy[None, ...])
                instance_patch_numpy = patch_data['A_instance'][0].detach().cpu().numpy()
                instance_patches.append(instance_patch_numpy[None, ...])
                if 'B_instance' in patch_data.keys():
                    real_instance_patch_numpy = patch_data['B_instance'][0].detach().cpu().numpy()
                    real_instance_patches.append(real_instance_patch_numpy[None,...])

                if not return_minimum:   
                    mean_image = np.mean(sample_patches, axis=0, keepdims=True)
                    mean_patches.append(mean_image)

                    input_patch_numpy = input_patch[0].detach().cpu().numpy()
                    input_patches.append(input_patch_numpy[None, ...])

                    if 'P' in data.keys():
                        point_patch_numpy = point_patch[0].detach().cpu().numpy()
                        point_patches.append(point_patch_numpy) #[None, ...])

        gen_patches = np.concatenate(gen_patches, axis=0)
        real_patches = np.concatenate(real_patches, axis=0)
        std_patches = np.concatenate(std_patches, axis=0)
        instance_patches = np.concatenate(instance_patches, axis=0)
        if 'B_instance' in data.keys():
            real_instance_patches = np.concatenate(real_instance_patches, axis=0)

        if not return_minimum:
            input_patches = np.concatenate(input_patches, axis=0)
            if 'P' in data.keys():
                point_patches = np.concatenate(point_patches, axis=0)
                print('point', point_patch.shape)
            mean_patches = np.concatenate(mean_patches, axis=0)
            #print(mean_patches.shape, instance_patches.shape)
            #(max_image, sample_times, 3, 256, 256) (max_image, 3, 256, 256) (max_image, 3, 256, 256)
        print(gen_patches.shape, real_patches.shape, std_patches.shape)

        return input_patches, gen_patches, real_patches, \
            std_patches, mean_patches, synthseg_patches, \
                point_patches, instance_patches, real_instance_patches

"""
name="tnbc_0_cyclegan_unalign_1masks"
export folder="MoNuSeg_random23"
export folder="MoNuSeg_random23_wpoints"

ckpt_dir="/home/mengwei/redwood_research/Points2Image/CycleGAN/checkpoints_0227/"
data_root="/home/mengwei/redwood_research/processed_data/${folder}.h5"
export name="v4_unalign_norm_Ga_resnet_Gb_resnet_cyclegan"  #cyclegan

python generate_datasets_overlap_patches_monuseg.py \
--train_opt_file ${ckpt_dir}/${name}/train_opt.txt \
--dataroot ${data_root}


"""

if __name__ == '__main__':
    #parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
    #parser.add_argument('--sample_times', type=int, default=1,
    #                    help=('sample times'))
    #parser.add_argument('--num_visualize', type=int, default=10,
    #                    help=('number of images for visualization and sanity check'))
    #parser.add_argument('--overlap', type=int, default=8,
    #                help=('step size for the overlapping patches'))
    #args = parser.parse_args()
    sample_times_options = [10]
    max_visualize = 20
    save_prefix = '0309'

    print('sample_times', sample_times_options)
    print('max_visualize', max_visualize)
    opt = TrainOptions().load_opt()   # get training options
    opt.batch_size = 1
    opt.crop_size = 1000
    opt.no_flip = True
    opt.serial_batches = True
    opt.preprocess = ''
    opt.checkpoints_dir = os.path.split(os.path.split(opt.train_opt_file)[0])[0]

    hd = h5py.File(os.path.join(opt.dataroot), 'r')
    opt.crop_size = 256
    model = create_model(opt)      # create a cyclegan model
    model.isTrain = False
    model.setup(opt)               # regular setup: load and print networks; create schedulers
    model.eval()

    save_subdir = os.path.join(model.save_dir, opt.dataroot.split('/')[-1].split('.')[0])
    if 'test' in opt.dataroot:
        overlap = 256
        get_item_fn = get_item_single_mask
    elif 'train' in opt.dataroot:
        overlap = 256   #24
        get_item_fn = get_item_single_mask
    elif 'random' in opt.dataroot:
        overlap = 256 # 128 #32
        get_item_fn = get_item_synth_mask
    else:
        raise NotImplementedError
    overwrite = not opt.no_overwrite

    os.makedirs(save_subdir, exist_ok=True)
    print('overlap', overlap)

    for sample_times in sample_times_options:    
        fake_images_all = list()
        instance_all = list()
        std_images_all = list()
        point_images_all = list()

        real_images_all = list()
        real_instance_all = list()

        mask_index = 0
        num_visualize = 0
        print('Overwrite results? ', overwrite)
        #if overwrite and os.path.exists(os.path.join(save_subdir, f'generated_step{overlap}_sample{sample_times}.npy')) :
        #    os.system(f'rm -r -v *step{overlap}_sample{sample_times}*')
        if os.path.exists(os.path.join(save_subdir, f'{save_prefix}generated_step{overlap}_sample{sample_times}.npy')) and opt.no_overwrite:
            print('Result exists. Skip')
            sys.exit()
        for i in range(hd['images'].shape[0]):  # inner loop within one epoch
            data_s = time.time()
            data = get_item_fn(hd, opt, i, mask_index=mask_index)
            data_e = time.time()
            print(i, 'data', data_e - data_s)

            model_s = time.time()
            input_patches, gen_patches, \
            real_patches, std_patches, \
            mean_patches, synthseg_patches, \
            point_patches, \
            instance_patches, real_instance_patches = run_inference(model, 
                                                    data,
                                                    1000, 256, 
                                                    overlap=overlap, 
                                                    sample_times=sample_times,
                                                    max_image=25000,
                                                    return_minimum= False) #num_visualize > max_visualize)
            model_e = time.time()
            print(i, 'model', model_e - model_s)
            patch_id = 0 #input_patches.shape[0]//2
            if num_visualize < max_visualize:
                num_visualize += 1
                img_list = list()
                img_title = list()
                cmaps = list()
                # 4 samples
                for npatch in [0,1,2,3]:
                    img_list += [np.transpose(0.5*(1+gen_patches[patch_id, min(npatch, sample_times-1)]), (1,2,0))]
                    img_title += [f'sample{min(npatch, sample_times)}']
                    cmaps += ['jet']
                # 4 inputs
                inst_mask = instance_patches[patch_id][0]
                print(np.min(inst_mask), np.max(inst_mask))
                color_mask = np.zeros(inst_mask.shape+(3,), dtype=np.uint8)
                for label in np.unique(inst_mask):
                    if label>0:
                        color_mask[inst_mask==label,:] = np.random.randint(255,size=(3,))
                img_list += [color_mask]  #np.transpose(0.5*(1+instance_patches[0]), (1,2,0))]
                img_title += ['input_mask']
                cmaps += ['jet']
                for nchannel in range(2):
                    img_list += [input_patches[patch_id, nchannel]]
                    img_title += [f'input_channel{npatch}']
                input_rgb = np.transpose(input_patches[patch_id], (1,2,0))
                #if not opt.norm_seg:
                #    input_rgb[..., 2] = 2*input_rgb[..., 2] - 1
                input_rgb = 0.5*(input_rgb + 1)
                img_list += [input_rgb]
                img_title += ['hv_seg']
                cmaps += ['jet', 'jet', 'gist_gray']

                # real image
                img_list += [np.transpose(0.5*(1+real_patches[patch_id]), (1,2,0))]
                if isinstance(point_patches, np.ndarray):
                    img_list += [point_patches[patch_id]] 
                else:
                    img_list+=[np.random.random(real_patches[0].shape[1:3])] #point_patches[0]]
                #[np.transpose(0.5*(1+real_patches[1]), (1,2,0))]
                img_list += [np.transpose(0.5*(1+mean_patches[patch_id]), (1,2,0))]
                img_list += [np.transpose(0.5*(1+std_patches[patch_id]), (1,2,0))]
                cmaps += ['jet', 'gist_gray', 'jet', 'jet']
                img_title += ['real1', 'point', 'mean', 'std']

                if opt.use_synthseg:
                    for npatch in range(4):
                        img_list += [0.5*(1+synthseg_patches[0,0,0])]
                        img_title += [f'synthseg{npatch}']
                        cmaps += ['jet']
                print(img_title)
                fig = create_group_fig(img_list=img_list, 
                                    cmaps=cmaps,
                                    titles=img_title,
                                    save_name=os.path.join(save_subdir,'%sstep%d_patch%d_sample%d_%d.svg'%(save_prefix, 
                                                                                                   overlap, patch_id,sample_times,i)),
                                    format='svg',
                                    dpi=200,
                                    adjust=True)
            
            assert(std_patches.shape == real_patches.shape)
            fake_images_all.append(gen_patches)
            real_images_all.append(real_patches)
            instance_all.append(instance_patches)
            real_instance_all.append(real_instance_patches)
            std_images_all.append(std_patches)
            point_images_all.append(point_patches)

        fake_images_all = np.concatenate(fake_images_all, axis=0)
        real_images_all = np.concatenate(real_images_all, axis=0)
        instance_all = np.concatenate(instance_all, axis=0)
        real_instance_all = np.concatenate(real_instance_all, axis=0)[:, None, ...]
        std_images_all  = np.concatenate(std_images_all, axis=0)
        point_images_all = np.concatenate(point_images_all, axis=0)

        npatches, nsamples, c, w, h = fake_images_all.shape
        generated_images = np.reshape(fake_images_all, (npatches*nsamples, c, w, h))
        npatches, ci, w, h = instance_all.shape
        np.save(os.path.join(save_subdir, f'{save_prefix}instance_single'), instance_all)
        np.save(os.path.join(save_subdir, f'{save_prefix}point_single'), point_images_all)

        instance_all = instance_all[:, None, ...]
        instance_all = np.repeat(instance_all, nsamples, axis=1)
        instance_all = np.reshape(instance_all, (npatches*nsamples, ci, w, h))
        
        #print(generated_images.shape, instance_all.shape)
        with h5py.File(os.path.join(save_subdir, f'{save_prefix}step{overlap}_sample{sample_times}.h5'), 'w') as h5f:
             h5f.create_dataset('images', data=fake_images_all)
             h5f.create_dataset('points', data=point_images_all)
             h5f.create_dataset('instance_masks', data=instance_all)

        ## Save for FID/KID 
        #np.save(os.path.join(save_subdir, f'{save_prefix}generated_step{overlap}_sample{sample_times}'), fake_images_all)
        #np.save(os.path.join(save_subdir, f'{save_prefix}generated_merge_step{overlap}_sample{sample_times}'), np.reshape(fake_images_all, (npatches*nsamples, c, w, h)))
        # np.save(os.path.join(save_subdir, f'{save_prefix}generated_single'), fake_images_all[:,0,...])
        # np.save(os.path.join(save_subdir, f'{save_prefix}real'), real_images_all)
        # np.save(os.path.join(save_subdir, f'{save_prefix}std'), std_images_all)

        ## Save for training segmentation
        # generated_images = np.transpose(generated_images, (0,2,3,1))
        # generated_images = np.clip((generated_images+1)*255/2.0, 0, 255).astype(np.uint8)
        # instance_all = np.transpose(instance_all, (0,2,3,1)).astype(np.uint16)
        # print(generated_images.shape, instance_all.shape)
        # with h5py.File(os.path.join(save_subdir, f'train_dataset_step{overlap}_sample{sample_times}.h5'), 'w') as h5f:
        #     h5f.create_dataset('images', data=generated_images)
        #     h5f.create_dataset('instance_masks', data=instance_all)
        # print('saved', os.path.join(save_subdir, f'train_dataset_step{overlap}_sample{sample_times}.h5'))
        # # sys.exit()

        # real_images_all = np.transpose(real_images_all, (0,2,3,1))
        # real_images_all = np.clip((real_images_all+1)*255/2.0, 0, 255).astype(np.uint8)
        # real_instance_all = np.transpose(real_instance_all, (0,2,3,1)).astype(np.uint16)
        # print(real_images_all.shape, real_instance_all.shape)
        # with h5py.File(os.path.join(save_subdir, 'real_train_dataset.h5'), 'w') as h5f:
        #     h5f.create_dataset('images', data=real_images_all)
        #     h5f.create_dataset('instance_masks', data=real_instance_all)
