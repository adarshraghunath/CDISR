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

def inverse_normalize_data_in_window(x_norm, w_min, w_max):
    '''Inverts the linear operation included in normalize_data_in_window. Cannot handle data clipping. :paramx_norm: Normalized image. :paramw_min: Lower bound of window. :paramw_max: Upper bound of window. :return: Tensor. ''' 
    return (x_norm * (w_max - w_min)) + w_min

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

class SinogramData(Dataset):
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

        self._f = self.filenames.copy()

        if test:

            self.hr_transforms = transform_lib.Compose([transform_lib.ToTensor()]) 
            
            
        else:
            self.hr_transforms = transform_lib.Compose(
                [
                    transform_lib.RandomCrop(hr_image_size),    
                    transform_lib.RandomVerticalFlip(p=0.5),
                    transform_lib.RandomHorizontalFlip(p=0.5),
                    transform_lib.ToTensor(),
                                                                 

                ]
            )

            
        


    def __getitem__(self, index):
        
        assert len(self.filenames) != 0
        filename = self.filenames[index]


        if self.image_type == '.tif':
            img = load_tiff_stack(self._f[index])
            if img.ndim==4:
                img = np.squeeze(img,axis = 0) 
                img = np.squeeze(img,axis=0)
            elif img.ndim==3:
                img = np.squeeze(img,axis=0)
            img = Image.fromarray(img)

        #Normalize and apply transformations

        img_hr = normalize_data_in_window(self.hr_transforms(img),-1.0,13.0) 
        _,H,W = img_hr.shape 
        img_lr = img_hr[:,::self.scale_factor,:]
        srtransf = transform_lib.Resize((H,W),Image.BICUBIC)
        img_sr = srtransf(img_lr)

        return {'LR': img_lr, 'HR': img_hr, 'SR': img_sr, 'Index': index}

    def __len__(self):
        return len(self.filenames)

