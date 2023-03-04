import importlib
import random
import os
import re

import cv2
import numpy as np
import torch

from dataset import get_dataset


class Config(object):
    """Configuration file."""

    def __init__(self):
        self.seed = 10

        self.logging = True

        # turn on debug flag to trace some parallel processing problems more easily
        self.debug = False

        self.model_name = "hovernet"
        self.model_mode = "original" # choose either `original` or `fast`

        if self.model_mode not in ["original", "fast"]:
            raise Exception("Must use either `original` or `fast` as model mode")

        self.nr_type = None # number of nuclear types (including background)

        # whether to predict the nuclear type, availability depending on dataset!
        self.type_classification = False

        # shape information - 
        # below config is for original mode. 
        # If original model mode is used, use [270,270] and [80,80] for act_shape and out_shape respectively
        # If fast model mode is used, use [256,256] and [164,164] for act_shape and out_shape respectively
        aug_shape = [540, 540] # patch shape used during augmentation (larger patch may have less border artefacts)
        train_act_shape = [256, 256] # patch shape used as input to network - central crop performed after augmentation
        train_out_shape = [256, 256] # patch shape at output of network
        valid_act_shape = [1000, 1000] # patch shape used as input to network - central crop performed after augmentation
        valid_out_shape = [1000, 1000] # patch shape at output of network
        # act_shape = [270, 270] # patch shape used as input to network - central crop performed after augmentation
        # out_shape = [80, 80] # patch shape at output of network

        # if model_mode == "original":
        #     if act_shape != [270,270] or out_shape != [80,80]:
        #         raise Exception("If using `original` mode, input shape must be [270,270] and output shape must be [80,80]")
        # if model_mode == "fast":
        #     if act_shape != [256,256] or out_shape != [164,164]:
        #         raise Exception("If using `fast` mode, input shape must be [256,256] and output shape must be [164,164]")

        self.dataset_name = "consep" # extracts dataset info from dataset.py
        self.log_dir = "logs/" # where checkpoints will be saved

        # paths to training and validation patches
        self.train_dir_list = [
            "/home/cj/Research/Points2Image/hover_net/dataset/training_data/fake_monuseg/fake_monuseg/train/540x540_164x164/"
            # "/home/cj/Research/Points2Image/hover_net/dataset/training_data/monuseg_Ga_resnetZ_Gb_resnetZ_bs1_v4/monuseg_Ga_resnetZ_Gb_resnetZ_bs1_v4/train/540x540_164x164/"
        ]
        self.valid_dir_list = [
            "/home/cj/Research/Points2Image/hover_net/dataset/training_data/gt_monuseg/gt_monuseg/valid/1000x1000_1x1/"
        ]

        self.shape_info = {
            "train": {"input_shape": train_act_shape, "mask_shape": train_out_shape,},
            "valid": {"input_shape": valid_act_shape, "mask_shape": valid_out_shape,},
        }

        # * parsing config to the running state and set up associated variables
        self.dataset = get_dataset(self.dataset_name)

        module = importlib.import_module(
            "models.%s.opt" % self.model_name
        )
        self.model_config = module.get_config(self.nr_type, self.model_mode)
        self.otf_opt = None
    
    def load_config_from_args(self, args):
        if args.train_dir is not None:
            print('change the original train_dir_list {0} to {1}'.format(self.train_dir_list, args.train_dir))
            self.train_dir_list = [args.train_dir]
            exp_name = args.train_dir.split('/')[-4]
            self.log_dir = os.path.join('./logs', exp_name)

        if args.valid_dir is not None:
            print('change the original valid_dir_list {0} to {1}'.format(self.valid_dir_list, args.valid_dir))
            self.valid_dir_list = [args.valid_dir]

        if args.otf is not None:
            self.otf_opt = load_opt_from_file(args.otf) 
            self.otf_opt.crop_size = self.shape_info["train"]["input_shape"][0]
            self.otf_opt.dataroot = args.otf_dataroot
            self.otf_opt.checkpoints_dir = os.path.split(os.path.split(self.otf_opt.train_opt_file)[0])[0]
            self.otf_opt.preprocess = ['affine', 'crop']
            self.log_dir = os.path.join('./logs', exp_name+'_otf')
        else:
            self.otf_opt = None
        module = importlib.import_module(
                "models.%s.opt" % self.model_name
            )
        self.model_config = module.get_config(self.nr_type, self.model_mode, otf_opt=self.otf_opt, epoch=args.epoch)
        self.log_dir = self.log_dir + '_{}'.format(args.epoch)
        
        
        if not os.path.exists(self.log_dir):
                os.mkdir(self.log_dir)
        

def load_opt_from_file(file):
    class opt_class:
        pass

    opt = opt_class()
    opt.isTrain = False
    opt.train_opt_file = file
    with open(opt.train_opt_file, "r") as f:
        for line in f:
            if line == '----------------- Options ---------------\n':
                pass
            elif line == '----------------- End -------------------\n':
                pass
            else:
                k = line[:25].strip()
                v = re.sub(r"\[default\: \S*\]", "", line[27:].strip(), flags=re.IGNORECASE).strip()
                if len(re.findall("\d", v))>0 and len(re.findall('[a-zA-Z_]', v))==0 and k != 'gpu_ids':
                    if '.' in v:
                        v = float(v)
                    else:
                        v = int(v)
                elif v.lower() == 'true':
                    v = True
                elif v.lower() == 'false':
                    v = False
                setattr(opt, k, v)

    if opt.suffix:
        suffix = ('_' + opt.suffix.format(**vars(opt))) if opt.suffix != '' else ''
        opt.name = opt.name + suffix

    # self.print_options(opt)

    # set gpu ids
    str_ids = opt.gpu_ids.split(',')
    opt.gpu_ids = []
    for str_id in str_ids:
        id = int(str_id)
        if id >= 0:
            opt.gpu_ids.append(id)
    if len(opt.gpu_ids) > 0:
        torch.cuda.set_device(opt.gpu_ids[0])

    if opt.max_dataset_size == "inf":
        opt.max_dataset_size = float("inf")
    return opt