import torch
import model as Model
from collections import OrderedDict
from model.sr3_modules.unet import UNet
from model.sr3_modules.diffusion import GaussianDiffusion
import torch.nn as nn
import model as Model
import core.logger as Logger
import argparse
import json
import glob
from dataloader import *
import numpy as np
from torchmetrics import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure, MeanSquaredError
import pandas as pd
import xlsxwriter

from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity

import os
import sys
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
print(parent_dir)
sys.path.append(parent_dir)

from PR_SRGAN.srgan_module import SRGAN   # SRGAN Reference method

device = 'cuda'
def set_device( x):
        if isinstance(x, dict):
            for key, item in x.items():
                if item is not None:
                    x[key] = item.to(device)
        elif isinstance(x, list):
            for item in x:
                if item is not None:
                    item = item.to(device)
        else:
            x = x.to(device)
        return x

def calculate_mean(numbers):
    if not numbers:
        return None
    return sum(numbers) / len(numbers)

def calculate_variance(numbers):
    if not numbers:
        return None
    mean = calculate_mean(numbers)
    squared_diff_sum = sum((x - mean) ** 2 for x in numbers)
    variance = squared_diff_sum / len(numbers)
    return variance

def load_tiff_stack(file):
    """ :paramfile: Path object describing the location of the file :return: a numpy array of the volume """

    if not (file.name.endswith('.tif') or file.name.endswith('.tiff')):
        raise FileNotFoundError('Input has to be tif.')
    else:
        volume = imread(file, plugin='tifffile')
        return volume

def save_to_tiff_stack(array, file):
    """ :paramarray: Array to save :paramfile: Path object to save to """

    if not file.parent.is_dir():
        file.parent.mkdir(parents=True, exist_ok=True)
    if not (file.name.endswith('.tif') or file.name.endswith('.tiff')):
        raise FileNotFoundError('File has to be tif.')
    else:
        imsave(file, array, plugin='tifffile', check_contrast=False)

def calc_lpips(x1,x2):
    x1 = x1.to(device)
    x2 = x2.to(device)
    noise = torch.cat((x1,)*3, dim=1)
    x_recon = torch.cat((x2,)*3, dim=1)

    return LPIPS(noise,x_recon)

model = UNet(
        in_channel=2,
        out_channel=1,
        norm_groups=32,
        inner_channel=64,
        channel_mults=[
                1,
                2,
                4,
                8,
                8
            ],
        attn_res=[16],
        res_blocks=2,
        dropout=0.2,
        image_size=512
    )

netG = GaussianDiffusion(
        model,
        image_size=512,
        channels=1,
        loss_type='l1',    # L1 or L2
        conditional=True,
        schedule_opt={
                "schedule": "linear",
                "n_timestep": 2000,
                "linear_start": 1e-6,
                "linear_end": 1e-2
            }
    )

PSNR = PeakSignalNoiseRatio(data_range = 1.0).to(device)
SSIM = StructuralSimilarityIndexMeasure(data_range = 1.0).to(device)

LPIPS = LearnedPerceptualImagePatchSimilarity(net_type='vgg',normalize=True).to(device)


schedule_opt = {
                "schedule": "linear",
                "n_timestep": 2000,
                "linear_start": 1e-6,
                "linear_end": 1e-2
            }
netG.set_loss(device)
netG.set_new_noise_schedule(schedule_opt, device)

#Image Transforms

hr_transforms = transform_lib.Compose([transform_lib.ToTensor()])

min_kernel_size = 3
max_kernel_size = 11

#set scale

scale = 4

bi_up = torch.nn.Upsample(scale_factor=scale, mode='bicubic')


# Diffusion Model Loading 

gen_path = "./checkpoint/_gen.pth"
state_dict = torch.load(gen_path)
network = set_device(netG)
network.load_state_dict(state_dict, strict=(True))
network.eval()


#SRGAN Load checkpoint

srgan = SRGAN.load_from_checkpoint(checkpoint_path="/.ckpt",map_location=torch.device('cuda'))

# Data Loading

test_data_path = "TestData/*.tif"

save_ = f'xrm_{scale}'

test_data = glob.glob(test_data_path)


#Evaluate

p = {'bi': [], 'diff': [],'srgan':[]}
s = {'bi': [], 'diff': [],'srgan':[]}
l= {'bi': [], 'diff': [],'srgan':[]}
c = 0
for i in range(len(test_data)):
    c+=1
    kernel_size = random.randint(min_kernel_size, max_kernel_size)
    scaled_kernel_size = max(kernel_size // 4, 1) * 2 + 1

    lr_transforms = transform_lib.Compose(
                [
                    transform_lib.Normalize(mean=(-1.0,) * 1, std=(2.0,) * 1),
                    transform_lib.Resize(256 // scale, Image.BICUBIC),
                ])
    
    img = load_tiff_stack(Path(test_data[i]))
    for _ in range(2):
        img = np.squeeze(img,axis =0)
        
    hr = Image.fromarray(img)

    hr = normalize_data_in_window(hr_transforms(hr),-1,1)

    hr = hr.to(device)

    lr = lr_transforms(hr)

    bicubic = bi_up(lr.unsqueeze(0))

    bicubic = torch.Tensor(bicubic).to(device)

    
    with torch.no_grad():
        srgan_op = srgan(lr.unsqueeze(0))
        SR = network.super_resolution(bicubic, True)
        
        
    
    diff_image = SR[-1]

    p['bi'].append(PSNR(bicubic, hr.unsqueeze(0)).item() )
    s['bi'].append(SSIM(bicubic,hr.unsqueeze(0)).item())
    l['bi'].append(calc_lpips((bicubic),(hr.unsqueeze(0))).item())

    p['srgan'].append(PSNR(srgan_op, hr).item())
    s['srgan'].append(SSIM(srgan_op,hr.unsqueeze(0)).item())
    l['srgan'].append(calc_lpips(srgan_op,hr.unsqueeze(0)).item())

    p['diff'].append(PSNR(diff_image,hr).item())
    s['diff'].append(SSIM(diff_image.unsqueeze(0),hr.unsqueeze(0)).item())
    l['diff'].append(calc_lpips(diff_image.unsqueeze(0),hr.unsqueeze(0)).item())



    hr_og = hr.cpu().detach().numpy()
    file_path_hr = Path(f'testdata/{save_}/' + str(c)+ '_hr.tif')
    save_to_tiff_stack(hr_og, file_path_hr)

    clr = lr.cpu().detach().numpy()
    file_path_lr = Path(f'testdata/{save_}/'+ str(c)+ '_lr.tif')
    save_to_tiff_stack(clr, file_path_lr)

    cs = srgan_op.cpu().detach().numpy()
    file_path_srgan = Path(f'testdata/{save_}/'+ str(c)+ '_srgan.tif')
    save_to_tiff_stack(cs, file_path_srgan)

    cd = diff_image.cpu().detach().numpy()
    file_path_diff = Path(f'testdata/{save_}/'+ str(c)+ '_diff.tif')
    save_to_tiff_stack(cd, file_path_diff)

    cb = bicubic.cpu().detach().numpy()
    file_path_bi = Path(f'testdata/{save_}/'+ str(c)+ '_bicubic.tif')
    save_to_tiff_stack(cb, file_path_bi)

def save_measurement(psnr,ssim,lpips):
    p_df = pd.DataFrame(psnr)
    s_df = pd.DataFrame(ssim)
    l_df = pd.DataFrame(lpips)
    print(p_df)
    print(s_df)
    print(l_df)

    # Rename columns to match the naming convention
    p_df.columns = ['p_' + col for col in p_df.columns]
    s_df.columns = ['s_' + col for col in s_df.columns]
    l_df.columns = ['l_' + col for col in l_df.columns]

    combined_df = pd.concat([p_df, s_df,l_df],axis=1)

    average_row = combined_df.mean()
    combined_df = combined_df.append(average_row, ignore_index=True)

    writer = pd.ExcelWriter(f'testdata/{save_}/measurement.xlsx', engine='xlsxwriter')

    # Write the combined DataFrame to a worksheet
    combined_df.to_excel(writer, sheet_name='Combined_values', index=True)
    writer.save()
    

save_measurement(p,s,l)

print(f"PSNR : Bicubic - {calculate_mean(p['bi'])} +- {calculate_variance(p['bi'])}, \
      Diffusion - {calculate_mean(p['diff'])} +- {calculate_variance(p['diff'])}, SRGAN - {calculate_mean(p['srgan'])} +-  {calculate_variance(p['srgan'])}") 
print(f"SSIM : Bicubic - {calculate_mean(s['bi'])} +- {calculate_variance(s['bi'])}, \
      Diffusion - {calculate_mean(s['diff'])} +- {calculate_variance(s['diff'])}, SRGAN - {calculate_mean(s['srgan'])} +-  {calculate_variance(s['srgan'])}") 
print(f"LPIPS : Bicubic - {calculate_mean(l['bi'])} +- {calculate_variance(l['bi'])},  \
      Diffusion -  {calculate_mean(l['diff'])} +- {calculate_variance(l['diff'])}, SRGAN - {calculate_mean(l['srgan'])} +-  {calculate_variance(l['srgan'])}") 


