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


def center_crop(data_dict, crop_size=1000):
    cropper = transforms.CenterCrop(crop_size)
    data_dict['A'] = cropper(data_dict['A'])
    data_dict['B'] = cropper(data_dict['B'])
    return data_dict


if __name__ == '__main__':
    opt = TrainOptions().load_opt()   # get training options
    opt.batch_size = 1
    if opt.netG_A == 'oasis_256':
        opt.crop_size = 896
    else:
        opt.crop_size = 1000
    opt.no_flip = True
    opt.serial_batches = True
    opt.preprocess = ''
    opt.dataroot = '/home/cj/Research/Points2Image_old/processed_data/MoNuSeg_train_v4_enhanced_pcorrected.h5'
    opt.checkpoints_dir = os.path.split(os.path.split(opt.train_opt_file)[0])[0]
    dataset = create_dataset(opt)  # create a dataset given opt.dataset_mode and other options
    with h5py.File(os.path.join(dataset.dataset.dir), 'r') as h5f_r:
        uncropped_fake_masks = torch.from_numpy(h5f_r['gen_instance_masks'][...,0].astype(np.int64))

    model = create_model(opt)      # create a cyclegan model
    model.isTrain = False
    model.setup(opt)               # regular setup: load and print networks; create schedulers
    model.eval()

    save_subdir = os.path.join(model.save_dir, 'generated_train_images')
    os.makedirs(save_subdir, exist_ok=True)
    fake_masks = None
    generated_images = None
    for i, data in enumerate(dataset):  # inner loop within one epoch
        # import pdb
        # pdb.set_trace()
        if opt.netG_A == 'oasis_256':
            cropper = transforms.CenterCrop(896)
            data['A'] = cropper(data['A'])
            data['B'] = cropper(data['B'])
            cropped_fake_masks = cropper(uncropped_fake_masks[:,None]).squeeze().numpy()
        model.set_input(data)         # unpack data from dataset and apply preprocessing
        # print(model.real_A.shape)
        with torch.no_grad():
            input_image = model.netG_A(model.real_A)  # G_A(A)
            target_mask = model.real_A
        # for img_in in input_image, data['A_paths']:
        input_image_numpy = np.clip((input_image[0].permute(1,2,0).detach().cpu().numpy()+1)*255/2.0, 0, 255).astype(np.uint8)
        # print(target_mask.max())
        target_mask_numpy = target_mask[0].permute(1,2,0).detach().cpu().numpy()
        # target_mask_numpy = process(target_mask_numpy)[0]
        # target_mask = (target_mask + 1.0) / 2.0

        if generated_images is None:
            generated_images = np.zeros((len(dataset), )+ input_image_numpy.shape)
            # generated_images = np.zeros((len(dataset), )+(opt.crop_size,opt.crop_size,3))
        # if fake_masks is None:
            # fake_masks = np.zeros((len(dataset), )+target_mask_numpy.shape)
        generated_images[data['A_paths'].item()] = input_image_numpy
        # fake_masks[data['A_paths'].item()] = target_mask_numpy

        '''sanity check'''
        fig, axes = plt.subplots(2,2,figsize=(10,10))
        axes[0, 0].imshow(input_image_numpy)
        # axes[1].imshow(target_mask_numpy, cmap='gray')
        axes[0,1].imshow(target_mask_numpy[...,2], cmap='gray')
        axes[1,0].imshow(target_mask_numpy[...,0], cmap='jet')
        axes[1,1].imshow(target_mask_numpy[...,1], cmap='jet')
        # for fig_i, img_i in enumerate([input_image_numpy, target_mask]):
            # ax
            # plt.imshow(image)
            # plt.axis('off')
        fig.savefig(os.path.join(save_subdir,'sample_%d.jpg'%(i)), dpi=200)
        plt.close()

        '''add the segmentation training below
        pred = net(input_image)
        loss = fn(pred, target_mask)
        ...
        '''
        # import pdb
        # pdb.set_trace()
        # print(generated_images.shape)
        # print(cropped_fake_masks.shape)
        assert(generated_images.shape[:3] == cropped_fake_masks.shape[:3])
    with h5py.File(os.path.join(save_subdir, 'train_dataset.h5'), 'w') as h5f:
        h5f.create_dataset('images', data=generated_images)
        h5f.create_dataset('instance_masks', data=cropped_fake_masks)

        # print('End of epoch %d / %d \t Time Taken: %d sec' % (epoch, opt.n_epochs + opt.n_epochs_decay, time.time() - epoch_start_time))
