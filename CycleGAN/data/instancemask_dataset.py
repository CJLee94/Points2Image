import os
from data.base_dataset import BaseDataset, get_transform
from skimage import color  # require skimage
from PIL import Image
import numpy as np
import imgaug as iaa
import cv2
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

        if self.opt.setup_augmentor:
            self.setup_augmentor(0,0)

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
        # A_img = torch.from_numpy(A_img)
        
        P_img = hd['points_masks'][index]
        P_img = P_img.astype(np.float32) / 255.
        # P_img = torch.from_numpy(P_img[None, ...])

        if not self.unalign:  # load aligned B_img
            b_index = index
        else:
            # to make sure we are loading unpaired data, resample the index
            b_index = torch.randint(0, self.__len__(), ()).numpy()            
        B_img = hd['images'][b_index].astype("uint8")
        # B_img = 2 * B_img.astype(np.float32) / 255. - 1.0
        # B_img = np.transpose(B_img, (2,0,1))
        # B_img = torch.from_numpy(B_img)

        if self.unalign:
            AP = self.transform_A(torch.cat((A_img,P_img)))
            A_instance, P = AP[:1], AP[-1]
            B = self.transform_B(B_img)
        else:
            # apply the same image transformation (A and B will have the same point annotation)
            AB = self.transform_AB(torch.cat((A_img, B_img, P_img), dim=0))
            A_instance, B, P = AB[:1], AB[1:1+3], AB[-1]

        A_instance = A_instance[0].numpy().astype(np.int32)
        if self.shape_augs is not None:
            shape_augs = self.shape_augs.to_deterministic()
            B_img = shape_augs.augment_image(B_img)
            A_img = shape_augs.augment_image(A_img)
            C_img = shape_augs.augment_image(C_img)

        if self.input_augs is not None:
            input_augs = self.input_augs.to_deterministic()
            B_img = input_augs.augment_image(B_img)
        
        # get hv map and binary mask
        A_masks = gen_targets(A_instance, crop_shape=A_instance.shape)
        if self.norm_seg:
            A_masks['np_map'] = 2 * A_masks['np_map'] - 1.
        #print(A_masks['hv_map'].shape, A_masks['np_map'].shape)  # crop x crop x 2,  crop x crop
        A = np.concatenate((A_masks['hv_map'], A_masks['np_map'][..., None]), axis=-1)
        A = np.transpose(A, (2,0,1))
        A = torch.from_numpy(A).float()
        return_dict = {'A': A, 'B': B, 'A_paths': index, 'B_paths': index}
        #if P is not None:
        return_dict.update({'P': P})
        return return_dict

    def __len__(self):
        """Return the total number of images in the dataset."""
        hd = h5py.File(self.dir, 'r')
        return hd['images'].shape[0]
    
    def setup_augmentor(self, worker_id, seed):
        self.augmentor = self.__get_augmentation(self.mode, seed)
        self.shape_augs = iaa.Sequential(self.augmentor[0])
        self.input_augs = iaa.Sequential(self.augmentor[1])
        self.id = self.id + worker_id
        return
        
    def __get_augmentation(self, mode, rng):
        if mode == "train":
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
                    self.input_shape[0], self.input_shape[1], position="uniform"
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
        elif mode == "valid" or mode == "test":
            shape_augs = [
                # set position to 'center' for center crop
                # else 'uniform' for random crop
                iaa.CropToFixedSize(
                    self.input_shape[0]//32*32, self.input_shape[1]//32*32, position="center"
                )
            ]
            input_augs = []

        return shape_augs, input_augs
    

def fix_mirror_padding(ann):
    """Deal with duplicated instances due to mirroring in interpolation
    during shape augmentation (scale, rotation etc.).
    
    """
    current_max_id = np.amax(ann)
    inst_list = list(np.unique(ann))
    inst_list.remove(0)  # 0 is background
    for inst_id in inst_list:
        inst_map = np.array(ann == inst_id, np.uint8)
        remapped_ids = measurements.label(inst_map)[0]
        remapped_ids[remapped_ids > 1] += current_max_id
        ann[remapped_ids > 1] = remapped_ids[remapped_ids > 1]
        current_max_id = np.amax(ann)
    return ann


####
def gaussian_blur(images, random_state, parents, hooks, max_ksize=3):
    """Apply Gaussian blur to input images."""
    img = images[0]  # aleju input batch as default (always=1 in our case)
    ksize = random_state.randint(0, max_ksize, size=(2,))
    ksize = tuple((ksize * 2 + 1).tolist())

    ret = cv2.GaussianBlur(
        img, ksize, sigmaX=0, sigmaY=0, borderType=cv2.BORDER_REPLICATE
    )
    ret = np.reshape(ret, img.shape)
    ret = ret.astype(np.uint8)
    return [ret]


####
def median_blur(images, random_state, parents, hooks, max_ksize=3):
    """Apply median blur to input images."""
    img = images[0]  # aleju input batch as default (always=1 in our case)
    ksize = random_state.randint(0, max_ksize)
    ksize = ksize * 2 + 1
    ret = cv2.medianBlur(img, ksize)
    ret = ret.astype(np.uint8)
    return [ret]


####
def add_to_hue(images, random_state, parents, hooks, range=None):
    """Perturbe the hue of input images."""
    img = images[0]  # aleju input batch as default (always=1 in our case)
    hue = random_state.uniform(*range)
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    if hsv.dtype.itemsize == 1:
        # OpenCV uses 0-179 for 8-bit images
        hsv[..., 0] = (hsv[..., 0] + hue) % 180
    else:
        # OpenCV uses 0-360 for floating point images
        hsv[..., 0] = (hsv[..., 0] + 2 * hue) % 360
    ret = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
    ret = ret.astype(np.uint8)
    return [ret]


####
def add_to_saturation(images, random_state, parents, hooks, range=None):
    """Perturbe the saturation of input images."""
    img = images[0]  # aleju input batch as default (always=1 in our case)
    value = 1 + random_state.uniform(*range)
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    ret = img * value + (gray * (1 - value))[:, :, np.newaxis]
    ret = np.clip(ret, 0, 255)
    ret = ret.astype(np.uint8)
    return [ret]


####
def add_to_contrast(images, random_state, parents, hooks, range=None):
    """Perturbe the contrast of input images."""
    img = images[0]  # aleju input batch as default (always=1 in our case)
    value = random_state.uniform(*range)
    mean = np.mean(img, axis=(0, 1), keepdims=True)
    ret = img * value + mean * (1 - value)
    ret = np.clip(img, 0, 255)
    ret = ret.astype(np.uint8)
    return [ret]


####
def add_to_brightness(images, random_state, parents, hooks, range=None):
    """Perturbe the brightness of input images."""
    img = images[0]  # aleju input batch as default (always=1 in our case)
    value = random_state.uniform(*range)
    ret = np.clip(img + value, 0, 255)
    ret = ret.astype(np.uint8)
    return [ret]
