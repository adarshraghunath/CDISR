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
from torch_radon import RadonFanbeam
from dataloader import *
from helper import *
import numpy as np
from torchmetrics import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure, MeanSquaredError
import matplotlib.pyplot as plt
import os

import pandas as pd
import xlsxwriter

from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity



import os
import sys
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
print(parent_dir)
sys.path.append(parent_dir)

from PR_SRGAN.srgan_module import SRGAN  # SRGAN Reference method

device = 'cuda'




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

# lr_transforms = transform_lib.Compose(
#                 [
#                     transform_lib.Normalize(mean=(-1.0,) * 1, std=(2.0,) * 1),
#                     # transform_lib.ToPILImage(),
#                     transform_lib.Resize(64, Image.BICUBIC),
# #                     transform_lib.ToTensor(),
#                 ])

# bi_up = torch.nn.Upsample(scale_factor=4, mode='bicubic')

def assignradon(scale,image_size,angles,vox_scaling,metadata,prj):
    angles = angles[::scale]
    radon = RadonFanbeam(image_size,
                        angles,
                        source_distance=vox_scaling * metadata['dso'],
                        det_distance=vox_scaling * metadata['ddo'],
                        det_count=prj.shape[1],
                        det_spacing=vox_scaling * metadata['du'],
                        clip_to_circle=False)
    return radon

def recon(sinogram,r,fn):
    reco = []
    with torch.no_grad():
        filtered_sinogram = r.filter_sinogram(sinogram, filter_name=filter_name) #This point

        # print(filtered_sinogram.shape)
        fbp = r.backprojection(filtered_sinogram)
        fbp = fbp.cpu().detach().numpy()
        
    reco.append(fbp)

    reco = np.array(reco)
    # # Scale reconstruction to HU values following the DICOM-CT-PD
    # # User Manual Version 3: WaterAttenuationCoefficient description
    fbp_hu = 1000 * ((reco - metadata['hu_factor']) / metadata['hu_factor'])

    print(fbp_hu.shape)

    plt.imsave('/home/woody/iwi5/iwi5119h/SinSR3/testresults/{}/{}.png'.format(save_,fn),fbp_hu[0], cmap='gray', vmin=-150, vmax=250)
    # plt.show()

    # save_to_tiff_stack(fbp_hu, Path('testresults/{}/{}.tif'.format(mpth,fn)))

    return fbp_hu

def norm(x):
    return (x - x.min()) / (x.max() - x.min())

def calc_lpips(x1,x2):
    x1 = x1.to(device)
    x2 = x2.to(device)
    noise = torch.cat((x1,)*3, dim=1)
    x_recon = torch.cat((x2,)*3, dim=1)

    return LPIPS(noise,x_recon)

# Model Loading 


gen_path = "/home/woody/iwi5/iwi5119h/SinSR3/experiments/sino_4_230822_173558/checkpoint/I140000_E700_gen.pth"
start_marker = "experiments/"
end_marker = "/checkpoint"

start_index = gen_path.find(start_marker) + len(start_marker)
end_index = gen_path.find(end_marker)

mpth = gen_path[start_index:end_index]


state_dict = torch.load(gen_path)
network = set_device(netG,device)
network.load_state_dict(state_dict, strict=(True))
network.eval()


srgan = SRGAN.load_from_checkpoint(
    checkpoint_path="/home/woody/iwi5/iwi5119h/PR_SRGAN/lightning_logs/srgan/Sinogram_4/checkpoints/epoch=999-step=400000.ckpt",map_location=torch.device('cuda'))


# Data Loading

# hr = "/home/woody/iwi5/iwi5119h/Test_Dataset/8.tif"
test_data_path = "/home/woody/iwi5/iwi5119h/Sinogram/SR_abdomen_projections/Test/"

files = os.listdir(test_data_path)
test_data = glob.glob(test_data_path)

p = {'bi': [], 'diff': [],'srgan':[]}
s = {'bi': [], 'diff': [],'srgan':[]}
l= {'bi': [], 'diff': [],'srgan':[]}

#Recon Algorithm specific 

# patient_id = 'L056'
dose_lv = 'hd'
slices= [28,107]
scale = 4
save_ = f'sino_new_{scale}'
fold = f'/home/woody/iwi5/iwi5119h/SinSR3/testresults/{save_}'
if not os.path.exists(fold):
    os.makedirs(fold)
c = 0
for j in files:

    name_prjs = '{}_{}_proj_fan_geometry'.format(j, dose_lv)  #replace patient_id with j
    prj_0, metadata = load_tiff_stack_with_metadata(Path('/home/woody/iwi5/iwi5119h/Sinogram/SR_abdomen_projections/Test/{}/{}.tif'.format(j, name_prjs))) 

    for i in slices:
        image_size = 512
        voxel_size = 0.7  # [mm]
        filter_name = "hann"
        # print(prj_0.shape)
        # for i in range(prj_0.shape[2]):
        prj = np.copy(np.flip(prj_0[:, :, i], axis=1)) #Use slice for i
        # prj = copy.deepcopy(np.expand_dims(prj, 1))
        # prj = copy.deepcopy(np.flip(prj.transpose([3, 1, 0, 2]), axis=2))
        angles = np.array(metadata['angles'])[:metadata['rotview']] + (np.pi / 2)   # [::4]
        
        vox_scaling = 1 / voxel_size

        radon = RadonFanbeam(image_size,
                                angles,
                                source_distance=vox_scaling * metadata['dso'],
                                det_distance=vox_scaling * metadata['ddo'],
                                det_count=prj.shape[1],
                                det_spacing=vox_scaling * metadata['du'],
                                clip_to_circle=False)
        
        

        radon_16 = assignradon(scale,image_size,angles,vox_scaling,metadata,prj)

        sinogram = torch.tensor(prj * vox_scaling).cuda()

        lr_sin = sinogram[::scale,...]

        # lr_srgan = sinogram[::scale,::scale]

        bicubic = torch.nn.Upsample(size=sinogram.shape, mode='bicubic')
        bicubic_sin = bicubic(lr_sin.unsqueeze(0).unsqueeze(0))

        with torch.no_grad():
            srgan_op = srgan(normalize_data_in_window(lr_sin.unsqueeze(0).unsqueeze(0),-1.0,13.0))
            SR = network.super_resolution(normalize_data_in_window(bicubic_sin,-1.0,13.0), True)
        
        SR = inverse_normalize_data_in_window(SR[-1],-1.0,13.0)
        srgan_op = inverse_normalize_data_in_window(srgan_op[:,:,:,::scale],-1.0,13.0)
        
        print(f"SRGAN- {srgan_op.shape}, Diff - {SR.shape}, HR - {sinogram.shape}, LR- {lr_sin.shape}, Bicu - {bicubic_sin.shape}")

# SRGAN- torch.Size([1, 1, 2304, 736]), Diff - torch.Size([1, 2304, 736]), HR - torch.Size([2304, 736]), LR- torch.Size([288, 736]), Bicu - torch.Size([1, 1, 2304, 736])
        
        im_lr = recon(lr_sin,radon_16,f"{j}_{str(i)}_lr")
        im_hr = torch.Tensor(recon(sinogram,radon,f"{j}_{str(i)}_hr"))  
        im_srgan = torch.Tensor(recon(srgan_op.squeeze(0).squeeze(0),radon,f"{j}_{str(i)}_srgan"))  
        im_sr = torch.Tensor(recon(SR.squeeze(0),radon,f"{j}_{str(i)}_sr"))  
        im_bi = torch.Tensor(recon(bicubic_sin.squeeze(0).squeeze(0),radon,f"{j}_{str(i)}_bi"))

        print(f"SRGAN- {im_srgan.shape}, Diff - {im_sr.shape}, HR - {im_hr.shape}, LR- {im_lr.shape}, Bicu - {im_bi.shape}")

        p['bi'].append(PSNR(norm(im_bi), norm(im_hr)).item() )
        s['bi'].append(SSIM(norm(im_bi.unsqueeze(0)),norm(im_hr.unsqueeze(0))).item())
        l['bi'].append(calc_lpips(norm(im_bi.unsqueeze(0)),norm(im_hr.unsqueeze(0))).item())

        p['srgan'].append(PSNR(norm(im_srgan), norm(im_hr)).item())
        s['srgan'].append(SSIM(norm(im_srgan.unsqueeze(0)),norm(im_hr.unsqueeze(0))).item())
        l['srgan'].append(calc_lpips(norm(im_srgan.unsqueeze(0)),norm(im_hr.unsqueeze(0))).item())

        p['diff'].append(PSNR(norm(im_sr),norm(im_hr)).item())
        s['diff'].append(SSIM(norm(im_sr.unsqueeze(0)),norm(im_hr.unsqueeze(0))).item())
        l['diff'].append(calc_lpips(norm(im_sr.unsqueeze(0)),norm(im_hr.unsqueeze(0))).item())
        

        # hr_og = im_hr.cpu().detach().numpy()
        # file_path_hr = Path(f'testdata/{save_}/' + str(c)+ '_hr.tif')
        # save_to_tiff_stack(hr_og, file_path_hr)

        # clr = im_lr.cpu().detach().numpy()
        # file_path_lr = Path(f'testdata/{save_}/'+ str(c)+ '_lr.tif')
        # save_to_tiff_stack(clr, file_path_lr)

        # cs = im_srgan.cpu().detach().numpy()
        # file_path_srgan = Path(f'testdata/{save_}/'+ str(c)+ '_srgan.tif')
        # save_to_tiff_stack(cs, file_path_srgan)

        # cd = im_sr.cpu().detach().numpy()
        # file_path_diff = Path(f'testdata/{save_}/'+ str(c)+ '_diff.tif')
        # save_to_tiff_stack(cd, file_path_diff)

        # cb = im_bi.cpu().detach().numpy()
        # file_path_bi = Path(f'testdata/{save_}/'+ str(c)+ '_bicubic.tif')
        # save_to_tiff_stack(cb, file_path_bi)

        # c+= 1



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

    print(combined_df)
    # combined_df.reset_index(drop=True, inplace=True)

    average_row = combined_df.mean()
    combined_df = combined_df.append(average_row, ignore_index=True)

    writer = pd.ExcelWriter(f'/home/woody/iwi5/iwi5119h/SinSR3/testresults/{save_}/measurement.xlsx', engine='xlsxwriter')

    # p_df.to_excel(writer, sheet_name='P_values', index=False)
    # s_df.to_excel(writer, sheet_name='S_values', index=False)

    

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

