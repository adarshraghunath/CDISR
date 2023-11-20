from pathlib import Path
import functools
import numpy as np
from skimage.io import imread, imsave
import random
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms as transform_lib
from torchvision.transforms import functional as TF
from multiprocessing import Process

IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP', '.tif'
]


def is_image(path: Path):
    return path.suffix in IMG_EXTENSIONS


def pad(img, scale):
    width, height = img.size
    pad_h = width % scale
    pad_v = height % scale
    img = TF.pad(img, (0, 0, scale - pad_h, scale - pad_v), padding_mode='reflect')
    return img

def normalize_data_in_window(x, w_min, w_max):
    '''Maps data in range [w_min, w_max] to [0, 1]. Clipping of data to zero or one for values outside of [w_min, w_max]. :paramx: Input image. :paramw_min: Lower bound of window. :paramw_max: Upper bound of window. :return: Normalized tensor. '''
    x_norm = (x - w_min) / (w_max - w_min)
    x_norm[x_norm >=1.0] =1.0 
    x_norm[x_norm <=0.0] =0.0 
    return x_norm

def load_tiff_stack(file):
    """ :paramfile: Path object describing the location of the file :return: a numpy array of the volume """

    if not (file.name.endswith('.tif') or file.name.endswith('.tiff')):
        raise FileNotFoundError('Input has to be tif.')
    else:
        volume = imread(file, plugin='tifffile')
        return volume

def ret_dl(dataset,bs):
   
   return  DataLoader(
                dataset,
                batch_size=bs,
                shuffle=True,
                num_workers=4,
                drop_last=True,
                pin_memory=True,
            )

class DatasetFromFolder(Dataset):
    def __init__(self, data_dir, scale_factor, image_type : str, image_channels: int, patch_size=256, n_workers=2, preupsample=False, training = True):
        assert patch_size % scale_factor == 0
        self.patch_size = patch_size
        if training:
            dataset = 'train'
            test = False
        else: 
            dataset = 'test'
            test = True    #Test loader, only for patches
        
        if isinstance(data_dir, str):
            data_dir = Path(data_dir)

        self.image_type = image_type
        print(self.image_type)
        if self.image_type == '.tif':
            filenames = []
            self.filenames = [f for f in data_dir.glob('*') if is_image(f)]

        else:
            self.filenames = [f for f in data_dir.glob('*') if is_image(f)]

        

        print("Number of images in %s dataset is %s with patchsize %s"%(dataset,len(self.filenames),self.patch_size))
        self.scale_factor = scale_factor
        self.preupsample = preupsample
        hr_image_size = self.patch_size
        lr_image_size = hr_image_size // scale_factor

        min_kernel_size = 3
        max_kernel_size = 11
        kernel_size = random.randint(min_kernel_size, max_kernel_size)
        scaled_kernel_size = max(kernel_size // 4, 1) * 2 + 1

        self._f = self.filenames.copy()

        if test:

            self.hr_transforms = transform_lib.Compose([transform_lib.ToTensor()]) 
            self.lr_transforms = transform_lib.Compose(
                [
                    transform_lib.Normalize(mean=(-1.0,) * image_channels, std=(2.0,) * image_channels),
                    transform_lib.Resize(lr_image_size, Image.BICUBIC),
                ]
            )
            
        else:
            self.hr_transforms = transform_lib.Compose(
                [
                    transform_lib.RandomCrop(hr_image_size),    #Change to RandomCrop later 
                    transform_lib.RandomVerticalFlip(p=0.5),
                    transform_lib.RandomHorizontalFlip(p=0.5),
                    transform_lib.ToTensor(),

                ]
            )

            self.lr_transforms = transform_lib.Compose(
                [

                    transform_lib.Normalize(mean=(-1.0,) * image_channels, std=(2.0,) * image_channels),
                    transform_lib.Resize(lr_image_size, Image.BICUBIC),
                ]
            )

        self.sr_transforms = transform_lib.Resize(hr_image_size,Image.BICUBIC)
                    


    def __getitem__(self, index):
        
        assert len(self.filenames) != 0
        filename = self.filenames[index]

        # img = Image.open(filename).convert('RGB')    #commented this out

        if self.image_type == '.tif':
            img = load_tiff_stack(self._f[index])
            if img.ndim==4:
                img = np.squeeze(img,axis = 0) 
                img = np.squeeze(img,axis=0)
            elif img.ndim==3:
                img = np.squeeze(img,axis=0)
            # print(img.shape)
            img = Image.fromarray(img)

        #Diff method
        img_hr = normalize_data_in_window(self.hr_transforms(img),-1,1)
        img_lr = self.lr_transforms(img_hr)
        img_sr = self.sr_transforms(img_lr)
        # return img_hr, img_lr

        return {'LR': img_lr, 'HR': img_hr, 'SR': img_sr, 'Index': index}

    def __len__(self):
        return len(self.filenames)




