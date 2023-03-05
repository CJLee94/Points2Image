import os
from data.base_dataset import BaseDataset, get_transform
from skimage import color  # require skimage
from PIL import Image
import numpy as np
import torchvision.transforms as transforms
import h5py
import torch
from util.targets import gen_targets


class InstanceMaskDataset(BaseDataset):
    """This dataset class can load a set of natural images in RGB, and convert RGB format into (L, ab) pairs in Lab color space.

    This dataset is required by pix2pix-based colorization model ('--model colorization')
    """
    @staticmethod
    def modify_commandline_options(parser, is_train):
        """Add new dataset-specific options, and rewrite default values for existing options.

        Parameters:
            parser          -- original option parser
            is_train (bool) -- whether training phase or test phase. You can use this flag to add training-specific or test-specific options.

        Returns:
            the modified parser.

        By default, the number of channels for input image  is 1 (L) and
        the number of channels for output image is 2 (ab). The direction is from A to B
        """
        parser.set_defaults(input_nc=3, output_nc=3, direction='AtoB')
        parser.add_argument('--num_masks', type=int, default=1, help='number of masks per point annotation')
        parser.add_argument('--norm_seg', action='store_true', default=False, help='if specified, normalize the binary segmentation from [0,1] to [-1,1]')
        parser.add_argument('--unalign', action='store_true', default=False, help='if specified, use unaligned pairs')

        return parser

    def __init__(self, opt):
        """Initialize this dataset class.

        Parameters:
            opt (Option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseDataset.__init__(self, opt)
        self.dir = os.path.join(opt.dataroot)
        print(self.dir)
        #assert(opt.input_nc == 1 and opt.output_nc == 3 and opt.direction == 'AtoB')
        #input_nc = self.opt.input_nc       # get the number of channels of input image
        #output_nc = self.opt.output_nc      # get the number of channels of output image
        
        self.num_masks = self.opt.num_masks
        self.norm_seg = self.opt.norm_seg
        self.unalign = self.opt.unalign
        if self.unalign:
            self.transform_A = get_transform(self.opt, grayscale=False)
            self.transform_B = get_transform(self.opt, grayscale=False)
        else:
            self.transform_AB = get_transform(self.opt, grayscale=False)
        
        print(f'number of mask per point annotation: {self.num_masks}')
        print(f'normalize segmentation? {self.norm_seg}')
        print(f'use unaligned pairs? {self.unalign}')

    def __getitem__(self, index):
        """Return a data point and its metadata information.

        Parameters:
            index - - a random integer for data indexing

        Returns a dictionary that contains A, B, A_paths and B_paths
            A (tensor) - - the source image
            B (tensor) - - the target image
            A_paths (str) - - image paths
            B_paths (str) - - image paths (same as A_paths)
        """
        hd = h5py.File(self.dir, 'r')
        # get the instance mask (scale 0 - N)
        A_img = hd['gen_instance_masks'][index]  # number of masks x W x H
        number_of_masks_per_images = self.num_masks #A_img.shape[0]
        assert number_of_masks_per_images <= A_img.shape[0]
        mask_index = 0
        if number_of_masks_per_images > 1:
            mask_index = torch.randint(0, number_of_masks_per_images, ()).numpy()
        A_img = A_img[mask_index: mask_index+1].astype(np.float32)  # 1 x W x H
        A_img = torch.from_numpy(A_img)

        B_img = hd['images'][index]
        B_img = 2 * B_img.astype(np.float32) / 255. - 1.0
        B_img = np.transpose(B_img, (2,0,1))
        B_img = torch.from_numpy(B_img)
        
        P_img = hd['points_masks'][index]
        P_img = P_img.astype(np.float32) / 255.
        P_img = torch.from_numpy(P_img[None, ...])

        if self.unalign:
            A_instance = self.transform_A(A_img)
            B = self.transform_B(B_img)
            P = None
        else:
            # apply the same image transformation (A and B will have the same point annotation)
            AB = self.transform_AB(torch.cat((A_img, B_img, P_img), dim=0))
            A_instance, B, P = AB[:1], AB[1:1+3], AB[-1]

        # get hv map and binary mask
        A_instance = A_instance[0].numpy().astype(np.int32)
        A_masks = gen_targets(A_instance, crop_shape=A_instance.shape)
        if self.norm_seg:
            A_masks['np_map'] = 2 * A_masks['np_map'] - 1.
        #print(A_masks['hv_map'].shape, A_masks['np_map'].shape)  # crop x crop x 2,  crop x crop
        A = np.concatenate((A_masks['hv_map'], A_masks['np_map'][..., None]), axis=-1)
        A = np.transpose(A, (2,0,1))
        A = torch.from_numpy(A).float()
        return_dict = {'A': A, 'B': B, 'A_paths': index, 'B_paths': index}
        if P is not None:
            return_dict.update({'P': P})
        return return_dict

    def __len__(self):
        """Return the total number of images in the dataset."""
        hd = h5py.File(self.dir, 'r')
        return hd['images'].shape[0]
