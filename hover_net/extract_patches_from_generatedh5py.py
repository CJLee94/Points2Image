"""extract_patches.py

Patch extraction script.
"""

import re
import glob
import os
import tqdm
import pathlib
import argparse

import numpy as np
import h5py

from misc.patch_extractor import PatchExtractor
from misc.utils import rm_n_mkdir

from dataset import get_dataset

# -------------------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', default=None)
    parser.add_argument('--valid', default=None)
    parser.add_argument('--test', default=None)
    parser.add_argument('--dataset_name', default='monuseg')
    parser.add_argument('--exp_name', default=None)
    args = parser.parse_args()
    # Determines whether to extract type map (only applicable to datasets with class labels).
    type_classification = False

    # Name of dataset - use Kumar, CPM17 or CoNSeP.
    # This used to get the specific dataset img and ann loading scheme from dataset.py
    exp_name = args.exp_name
    dataset_name = args.dataset_name+"_{}".format(exp_name)
    save_root = "dataset/"

    # a dictionary to specify where the dataset path should be
    dataset_info = dict()
    if args.train is not None:
        dataset_info["train"] = args.train
    if args.valid is not None:
        dataset_info["valid"] = args.valid
    if args.test is not None:
        dataset_info["test"] = args.test
    # dataset_info = {
        # "train": '/home/cj/Archive/CycleGAN_P0_checkpoints/{}/generated_train_images/train_dataset_x10.h5'.format(exp_name),
        # "valid": '/home/cj/Research/Points2Image_old/processed_data/MoNuSeg_v4_val_split.h5',
        # "test": '/home/cj/Research/Points2Image_old/processed_data/MoNuSeg_test_v4_enhanced_pcorrected.h5'
    # }

    patterning = lambda x: re.sub("([\[\]])", "[\\1]", x)
    # parser = get_dataset(dataset_name)
    for split_name, split_desc in dataset_info.items():
        if split_name == 'train':
            if 'tnbc' in args.dataset_name.lower():
                win_size = [270, 270]
                step_size = [82, 82]
            elif 'monuseg' in args.dataset_name.lower():
                win_size = [540, 540]
                step_size = [164, 164]
            extract_type = "valid"  # Choose 'mirror' or 'valid'. 'mirror'- use padding at borders. 'valid'- only extract from valid regions.
            xtractor = PatchExtractor(win_size, step_size)
        elif split_name == 'valid' or split_name == 'test':
            if 'tnbc' in args.dataset_name.lower():
                win_size = [512, 512]
            elif 'monuseg' in args.dataset_name.lower():
                win_size = [1000, 1000]
            step_size = [1, 1]
            extract_type = "valid"  # Choose 'mirror' or 'valid'. 'mirror'- use padding at borders. 'valid'- only extract from valid regions.
            xtractor = PatchExtractor(win_size, step_size)

        h5f = h5py.File(split_desc, 'r')
        # img_ext, img_dir = split_desc["img"]
        # ann_ext, ann_dir = split_desc["ann"]

        out_dir = "%s/%s/%s/%dx%d_%dx%d/" % (
            save_root,
            dataset_name,
            split_name,
            win_size[0],
            win_size[1],
            step_size[0],
            step_size[1],
        )
        # file_list = glob.glob(patterning("%s/*%s" % (ann_dir, ann_ext)))
        # file_list.sort()  # ensure same ordering across platform
        if split_name == 'train':
            image_stacks = h5f['images']
            ann_stacks = h5f['instance_masks']
        elif split_name == 'valid' or split_name == 'test':
            image_stacks = h5f['images']
            ann_stacks = h5f['instance_masks']
        # import pdb
        # pdb.set_trace()

        rm_n_mkdir(out_dir)

        pbar_format = "Process File: |{bar}| {n_fmt}/{total_fmt}[{elapsed}<{remaining},{rate_fmt}]"
        pbarx = tqdm.tqdm(
            total=ann_stacks.shape[0], bar_format=pbar_format, ascii=True, position=0
        )

        for file_idx, (img, ann) in enumerate(zip(image_stacks, ann_stacks)):
            # base_name = pathlib.Path(file_path).stem

            # img = parser.load_img("%s/%s%s" % (img_dir, base_name, img_ext))
            # ann = parser.load_ann(
                # "%s/%s%s" % (ann_dir, base_name, ann_ext), type_classification
            # )

            # *
            img = np.concatenate([img, ann[..., None]], axis=-1)
            sub_patches = xtractor.extract(img, extract_type)

            pbar_format = "Extracting  : |{bar}| {n_fmt}/{total_fmt}[{elapsed}<{remaining},{rate_fmt}]"
            pbar = tqdm.tqdm(
                total=len(sub_patches),
                leave=False,
                bar_format=pbar_format,
                ascii=True,
                position=1,
            )

            for idx, patch in enumerate(sub_patches):
                np.save("{0}/{1}_{2:03d}.npy".format(out_dir, 'file_'+str(file_idx), idx), patch)
                pbar.update()
            pbar.close()
            # *

            pbarx.update()
        pbarx.close()
