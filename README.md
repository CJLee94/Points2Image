# Microscopy Image Segmentation via Point and Shape Regularized Data Synthesis

This repository contains the official Python implementation of the paper "Microscopy Image Segmentation via Point and Shape Regularized Data Synthesis," accepted at the MICCAI DALI 2023 workshop.

## Overview

Our work introduces a novel approach to microscopy image segmentation, leveraging advanced techniques in deep learning to improve the accuracy and efficiency of segmentation models. The implementation consists of two major modules:

1. **CycleGAN Module**: Utilized to train a segmentation mask-conditioned image generator from image and point annotation pairs.
2. **HoVerNet Module**: Employed to train a segmentation model using the synthetic image and mask pairs generated by the CycleGAN.

### CycleGAN Module

The CycleGAN module is designed to create synthetic image and segmentation mask pairs. These pairs are generated from input images with point annotations, facilitating the training of segmentation models without the need for extensive hand-annotated segmentation masks.

### HoVerNet Module

After generating synthetic image-mask pairs using the CycleGAN module, the HoVerNet module is used for the actual segmentation task. This module can be adapted to train other segmentation models as well.

## Installation

Instructions for setting up the environment and installing necessary dependencies.

```bash
# Clone the repository
git clone https://github.com/CJLee94/Points2Image.git
cd Points2Image

# Install dependencies
pip install -r requirements.txt
```

## Usage

Detailed instructions on how to use each module, including training and inference processes.

### CycleGAN Module

```bash
# To train the CycleGAN model
python train.py --dataroot ~/redwood_research/processed_data/MoNuSeg_train_v4_enhanced.h5 \
--dataset_mode instancemask --name oasis_cyclegan_hv_map --model instancecyclegan \
--netG oasis_256 \
--pool_size 50 --no_dropout --phase train --preprocess crop \
--crop_size 256 --batch_size 1 --lambda_identity 0 --n_epochs 500 \
--n_epochs_decay 300 --input_nc 3 --setup_augmentor

# To generate synthetic image-mask pairs
python generate_datasets.py --train_opt_file /home/cj/Research/Points2Image/CycleGAN/checkpoints/basic_netD_oasis_netGa_unet256_netGb_cyclegan
```

### HoVerNet Module

```bash
# To train the HoVerNet model
python run_train.py --train_dir [path to generated training data] --valid_dir [path to generated validation data]
# To perform segmentation on new images
python hovernet_infer.py --options
```

## Dataset

TODO: Information about the dataset used, how to access it, and how to prepare it for training.

## Results

TODO: Brief overview of the results obtained with this approach, possibly including images or charts.

## Citation

If you find our work useful in your research, please consider citing:

```
Li, S., Ren, M., Ach, T., & Gerig, G. (2023). Microscopy Image Segmentation via Point and Shape Regularized Data Synthesis. arXiv preprint arXiv:2308.09835.
```

## Contact

For any queries regarding the code or the paper, feel free to reach out:

- Email: shijie.li@nyu.edu
- Institution: New York University, Tandon School of Engineering
