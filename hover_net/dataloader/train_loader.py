import csv
import glob
import os
import re

import cv2
import matplotlib.pyplot as plt
import numpy as np
import scipy.io as sio
import torch.utils.data

import imgaug as ia
from imgaug import augmenters as iaa
from misc.utils import cropping_center

from generative_models import create_model as create_generator
from .augs import (
    add_to_brightness,
    add_to_contrast,
    add_to_hue,
    add_to_saturation,
    gaussian_blur,
    median_blur,
)


####
def make_generator(opt):
    generator = create_generator(opt)      # create a cyclegan model
    generator.isTrain = False
    generator.setup(opt)               # regular setup: load and print networks; create schedulers
    generator.eval()
    return generator

####
class Augmenter:
    def __init__(self, 
                 mode = 'train',
                 otf = False,
                 otf_opt = None,
                 precrop_shape=512,
                 input_shape=256, 
                 target_gen=None,) -> None:
        self.otf = otf
        if self.otf:
            self.generator = make_generator(otf_opt)
        else:
            self.generator = None
        self.mode = mode
        self.precrop_shape = precrop_shape
        self.input_shape = input_shape

        self.target_gen_func = target_gen[0]
        self.target_gen_kwargs = target_gen[1]

        self.precrop_dict = dict()
        self.shape_aug_dict = dict()
        self.input_aug_dict = dict()
    
    def __call__(self, databatch):
        img, ann, worker_id = databatch
        return self.process_aug(img, ann, worker_id)

    def process_aug(self, img, ann, worker_id):
        img = img.numpy()
        ann = ann.numpy()
        worker_id = worker_id[0].item()

        # nphv_map = np.empty(ann.shape+(3,), dtype=np.float32)
        # feed_dict_list = []
        precrop_func, shape_aug_func, input_aug_func = self.get_augmenter(worker_id=worker_id)
        if self.otf and self.mode=="train":
            precrop = precrop_func.to_deterministic()
            ann = np.stack(precrop.augment_images(ann))

            nphv_map = torch.zeros(ann.shape[:3]+(3,))
            # import pdb
            # pdb.set_trace()
            for ann_id, ann_single in enumerate(ann):
                inst_map = ann_single[..., 0]  # HW1 -> HW
                target_dict = self.target_gen_func(
                    inst_map, [self.precrop_shape, self.precrop_shape], **self.target_gen_kwargs
                )
                # target_dict_list.append(target_dict)
                nphv=np.concatenate([target_dict["hv_map"].astype("float32"), 
                                     target_dict["np_map"][...,None].astype("float32")], -1)
                nphv_map[ann_id] = torch.from_numpy(nphv)
            # import pdb
            # pdb.set_trace()
            nphv_map = nphv_map.permute(0,3,1,2).contiguous()
            with torch.no_grad():
                img = self.generator.netG_A(nphv_map.to("cuda"))
            img = torch.clamp(255.0*(img+1)/2.0, 0, 255).detach().cpu().numpy().astype(np.uint8)
            img = img.transpose(0,2,3,1)

        # import pdb
        # pdb.set_trace()
        shape_aug = shape_aug_func.to_deterministic()
        img = np.stack(shape_aug.augment_images(img))
        ann = np.stack(shape_aug.augment_images(ann))

        input_aug = input_aug_func.to_deterministic()
        img = np.stack(input_aug.augment_images(img))

        feed_dict = {"img": torch.from_numpy(img)}
        target_dict_list = []
        for ann_single in ann:
            inst_map = ann_single[..., 0]  # HW1 -> HW
            target_dict = self.target_gen_func(
                inst_map, self.input_shape, **self.target_gen_kwargs
            )
            target_dict_list.append(target_dict)
        
        feed_dict['np_map'] = torch.stack([torch.from_numpy(d['np_map']) for d in target_dict_list],0)
        feed_dict['hv_map'] = torch.stack([torch.from_numpy(d['hv_map']) for d in target_dict_list],0)
        # if self.otf:
        #     precrop = precrop_func.to_deterministic()
        #     img_single = precrop.augment_image(img_single)
        #     ann_single = precrop.augment_image(ann_single)

        #     inst_map = ann_single[..., 0]  # HW1 -> HW

        #     target_dict = self.target_gen_func(
        #         inst_map, self.precrop_shape, **self.target_gen_kwargs
        #     )
        #     nphv=np.concatenate([target_dict["hv_map"].astype("float32"), 
        #                             target_dict["np_map"][...,None].astype("float32")], -1)

        #     nphv = torch.from_numpy(nphv[None]).permute(0,3,1,2).contiguous()
        #     img = self.generator(nphv.to("cuda"))
        #     img = torch.clamp(255.0*(img+1)/2.0, 0, 255)[0].detach().cpu().numpy().astype(np.uint8)
                
            
            
            
        #     # if self.with_type:
        #     #     type_map_single = (ann_single[..., 1]).copy()
        #     #     type_map_single = cropping_center(type_map_single, self.mask_shape)
        #     #     #type_map[type_map == 5] = 1  # merge neoplastic and non-neoplastic
        #     #     feed_dict["tp_map"] = type_map_single

        #     # TODO: document hard coded assumption about #input
        #     target_dict = self.target_gen_func(
        #         inst_map_single, self.input_shape, **self.target_gen_kwargs
        #     )
        #     feed_dict.update(target_dict)
        #     feed_dict_list.append(feed_dict)

        
        return feed_dict


    def get_augmenter(self, worker_id):
        if worker_id not in self.precrop_dict.keys():
            worker_seed = torch.randint(0, 2 ** 32, (1,))[0].cpu().item() + worker_id
            augmenter = self.__get_augmentation(self.mode, worker_seed)
            self.precrop_dict[worker_id] = iaa.Sequential(augmenter[0])
            self.shape_aug_dict[worker_id] = iaa.Sequential(augmenter[1])
            self.input_aug_dict[worker_id] = iaa.Sequential(augmenter[2])

        return self.precrop_dict[worker_id],self.shape_aug_dict[worker_id],self.input_aug_dict[worker_id]

    def __get_augmentation(self, mode, rng):
        if mode == "train":
            precrop = [
                iaa.CropToFixedSize(
                    384, 384, position="uniform", seed=rng
                ),
            ]

            shape_augs = [
                # * order = ``0`` -> ``cv2.INTER_NEAREST``
                # * order = ``1`` -> ``cv2.INTER_LINEAR``
                # * order = ``2`` -> ``cv2.INTER_CUBIC``
                # * order = ``3`` -> ``cv2.INTER_CUBIC``
                # * order = ``4`` -> ``cv2.INTER_CUBIC``
                # ! for pannuke v0, no rotation or translation, just flip to avoid mirror padding
                iaa.Affine(
                    # scale images to 80-120% of their size, individually per axis
                    scale={"x": (0.8, 1.2), "y": (0.8, 1.2)},
                    # translate by -A to +A percent (per axis)
                    translate_percent={"x": (-0.01, 0.01), "y": (-0.01, 0.01)},
                    shear=(-5, 5),  # shear by -5 to +5 degrees
                    rotate=(-179, 179),  # rotate by -179 to +179 degrees
                    order=0,  # use nearest neighbour
                    backend="cv2",  # opencv for fast processing
                    seed=rng,
                ),
                # set position to 'center' for center crop
                # else 'uniform' for random crop
                iaa.CropToFixedSize(
                    self.input_shape[0], self.input_shape[1], position="uniform", seed=rng,
                ),
                iaa.Fliplr(0.5, seed=rng),
                iaa.Flipud(0.5, seed=rng),
            ]

            input_augs = [
                iaa.OneOf(
                    [
                        iaa.Lambda(
                            seed=rng,
                            func_images=lambda *args: gaussian_blur(*args, max_ksize=3),
                        ),
                        iaa.Lambda(
                            seed=rng,
                            func_images=lambda *args: median_blur(*args, max_ksize=3),
                        ),
                        iaa.AdditiveGaussianNoise(
                            loc=0, scale=(0.0, 0.05 * 255), per_channel=0.5
                        ),
                    ]
                ),
                iaa.Sequential(
                    [
                        iaa.Lambda(
                            seed=rng,
                            func_images=lambda *args: add_to_hue(*args, range=(-8, 8)),
                        ),
                        iaa.Lambda(
                            seed=rng,
                            func_images=lambda *args: add_to_saturation(
                                *args, range=(-0.2, 0.2)
                            ),
                        ),
                        iaa.Lambda(
                            seed=rng,
                            func_images=lambda *args: add_to_brightness(
                                *args, range=(-26, 26)
                            ),
                        ),
                        iaa.Lambda(
                            seed=rng,
                            func_images=lambda *args: add_to_contrast(
                                *args, range=(0.75, 1.25)
                            ),
                        ),
                    ],
                    random_order=True,
                ),
            ]
        elif mode == "valid":
            precrop = []
            shape_augs = [
                # set position to 'center' for center crop
                # else 'uniform' for random crop
                iaa.CropToFixedSize(
                    self.input_shape[0], self.input_shape[1], position="center"
                )
            ]
            input_augs = []

        return precrop, shape_augs, input_augs

####
class FileLoader(torch.utils.data.Dataset):
    """Data Loader. Loads images from a file list and 
    performs augmentation with the albumentation library.
    After augmentation, horizontal and vertical maps are 
    generated.

    Args:
        file_list: list of filenames to load
        input_shape: shape of the input [h,w] - defined in config.py
        mask_shape: shape of the output [h,w] - defined in config.py
        mode: 'train' or 'valid'
        
    """

    # TODO: doc string

    def __init__(
        self,
        file_list,
        with_type=False,
        input_shape=None,
        mask_shape=None,
        mode="train",
        setup_augmentor=True,
        target_gen=None,
    ):
        assert input_shape is not None and mask_shape is not None
        self.worker_id = None
        self.worker_seed = None
        self.mode = mode
        self.info_list = file_list
        self.with_type = with_type
        self.mask_shape = mask_shape
        self.input_shape = input_shape
        self.id = 0
        self.target_gen_func = target_gen[0]
        self.target_gen_kwargs = target_gen[1]
        if setup_augmentor:
            self.setup_augmentor(0, 0)
        return

    # def setup_augmentor(self, worker_id, seed):
    #     self.worker_id = worker_id
    #     self.worker_seed = seed
    #     self.augmentor = self.__get_augmentation(self.mode, seed)
    #     self.precrop = iaa.Sequential(self.augmentor[0])
    #     self.shape_augs = iaa.Sequential(self.augmentor[1])
    #     self.input_augs = iaa.Sequential(self.augmentor[2])
    #     self.id = self.id + worker_id
    #     return

    def __len__(self):
        return len(self.info_list)

    def __getitem__(self, idx):
        path = self.info_list[idx]
        data = np.load(path)

        # split stacked channel into image and label
        # print(data[...,:3].max())
        img = (data[..., :3]).astype("uint8")  # RGB images
        ann = (data[..., 3:]).astype("int32")  # instance ID map and type map

        worker_id = torch.utils.data.get_worker_info().id

        return img, ann, worker_id
    
    # def pack_data(self, img, ann):
    #     if self.precrop is not None:
    #         precrop = self.precrop.to_deterministic()
    #         img = precrop.augment_image(img)
    #         ann = precrop.augment_image(ann)


    #     if self.shape_augs is not None:
    #         shape_augs = self.shape_augs.to_deterministic()
    #         img = shape_augs.augment_image(img)
    #         ann = shape_augs.augment_image(ann)

    #     if self.input_augs is not None:
    #         input_augs = self.input_augs.to_deterministic()
    #         img = input_augs.augment_image(img)

    #     img = cropping_center(img, self.input_shape)
    #     feed_dict = {"img": img}

    #     if len(ann.shape)==3 and ann.shape[-1]==3:
    #         inst_map = ann
    #     else:
    #         inst_map = ann[..., 0]  # HW1 -> HW
        
    #     if self.with_type:
    #         type_map = (ann[..., 1]).copy()
    #         type_map = cropping_center(type_map, self.mask_shape)
    #         #type_map[type_map == 5] = 1  # merge neoplastic and non-neoplastic
    #         feed_dict["tp_map"] = type_map

    #     # TODO: document hard coded assumption about #input
    #     target_dict = self.target_gen_func(
    #         inst_map, self.mask_shape, **self.target_gen_kwargs
    #     )
    #     feed_dict.update(target_dict)
    #     feed_dict["worker_id"] = self.worker_id
    #     feed_dict["worker_seed"] = self.worker_seed
    #     return feed_dict