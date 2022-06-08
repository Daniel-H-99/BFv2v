import re
import matplotlib

matplotlib.use('Agg')
import math
import os, sys
import glob
import yaml
from argparse import ArgumentParser
from time import gmtime, strftime
from shutil import copy

from frames_dataset import SingleImageDataset, SingleVideoDataset

import torch
import warnings

warnings.filterwarnings(action='ignore')

from tqdm import trange
from tqdm import tqdm
import torch
import numpy as np

from torch.utils.data import DataLoader

from logger import Logger

from frames_dataset import DatasetRepeater
from sync_batchnorm import DataParallelWithCallback

from utils.util import draw_section
from utils.one_euro_filter import OneEuroFilter
import imageio

import torch.nn as nn
from modules.headmodel import HeadModel
from modules.keypoint_detector import HEEstimator

def filter_values(values):
    MIN_CUTOFF = 0.001
    BETA = 0.07
    num_frames = len(values)
    fps = 25
    times = np.linspace(0, num_frames / fps, num_frames)
    
    filtered_values= []
    
    values = values
    for i, x in enumerate(values):
        if i == 0:
            filter_value = OneEuroFilter(times[0], x, min_cutoff=MIN_CUTOFF, beta=BETA)
        else:
            x = filter_value(times[i], x)
        
        filtered_values.append(x)
        
    res = np.array(filtered_values)
    
    res = res
    return res

def denormalize(he_estimator, mesh, img, reference_info, src_value=None, bias=None, is_src=False):
        # target pose
        print(f'img shape {img.shape}')
        with torch.no_grad():
            he = he_estimator(img) # {R, t, ...}
        R, t = he['R'][0], he['t'][0]
        value = mesh['he_value'] # N x 3
        aux = reference_info
        if bias is None:
            bias = mesh['t'].squeeze(1) + mesh['c'] * mesh['R'] @ t.detach().cpu() # 3
            bias = bias.detach().cpu()
        else:
            bias = mesh['c'] * aux['he_t'].detach().cpu() # 3
        mesh['he_bias'] = bias
        centered_value = value.cuda() - bias.unsqueeze(0).cuda()  # N x 3
        if not is_src:
            R_tilda = R # 3 x 3
        else:
            R_tilda = mesh['R'].inverse()
        # t = torch.cat([t[:2], t[[2]] - 0.1], dim=0)
        mesh['he_R'], mesh['he_t'] = R_tilda.detach().cpu(), t.detach().cpu()
        frontalized_value = (1 / mesh['c']) * torch.einsum('mp,kp->km', R_tilda.cuda(), centered_value)
        trans_value = frontalized_value + t[None]
        mesh['he_raw_value'] = trans_value.detach().cpu()

        if src_value is not None:
            centered_src_value = src_value.cuda() - bias.unsqueeze(0).cuda()
            frontalized_src_value = (1 / mesh['c']) * torch.einsum('mp,kp->km', R_tilda.cuda(), centered_src_value)
            trans_src_value = frontalized_src_value + t[None]
            mesh['fake_raw_value'] = trans_src_value.detach().cpu()
        
        
            
def main(config, model, res_dir, src_dataset, drv_dataset, threshold=None, he_estimator=None, reference_dataset=None):
    train_params = config['train_params']    
    section_indices = []
    for sec in model.module.sections:
        section_indices.extend(sec[0])
        
    src_dataloader = DataLoader(src_dataset, batch_size=1, shuffle=False, num_workers=0, drop_last=False)
    drv_sample_dataloader = DataLoader(drv_dataset, batch_size=(10 * train_params['batch_size']), shuffle=False, num_workers=0, drop_last=False)
    drv_dataloader = DataLoader(drv_dataset, batch_size=1, shuffle=False, num_workers=0, drop_last=False)
    
    src_data = iter(src_dataloader).next()
    drv_data = iter(drv_sample_dataloader).next()
    
    # load reference info
    ref_img_data = iter(DataLoader(reference_dataset, batch_size=1, shuffle=False, num_workers=0, drop_last=False)).next()
    with torch.no_grad():
        he_ref = he_estimator(ref_img_data['frame'].cuda())
    reference_info = {'R': ref_img_data['mesh']['R'][0].cuda(), 't': ref_img_data['mesh']['t'][0].cuda(), 'c': ref_img_data['mesh']['c'][0].cuda(), 'he_R': he_ref['R'][0].cuda(), 'he_t': he_ref['t'][0].cuda(), 'img': ref_img_data['frame'][0].permute(1, 2, 0).cuda()}
    
    with torch.no_grad():
        params_drv = model.module.estimate_params(drv_data)  # x: (num_sections) x N * 3, e: (num_sections) x B x N * 3, k_e: (num_sections) x N * 3
        x_drv, e_drv, k_e_drv = params_drv['x'], params_drv['e'], params_drv['k_e']
        # print(f'e_drv shape: {e_drv[0].shape}')
        # print(f'k_drv shape: {k_e_drv[0].shape}')
        
        e_normalized = [e_drv_sec / k_e_drv[i][None] for i, e_drv_sec in enumerate(e_drv)]
        # print(f'e_normed shape: {e_normalized[0].shape}')
        params_src = model.module.estimate_params_v3(src_data, drv_data, e_normalized, threshold=threshold)
        # params_src = model.module.estimate_params(src_data)
        # params_src = model.module.estimate_params(src_data)  # x: (num_sections) x N * 3, e: (num_sections) x B x N * 3, k_e: (num_sections) x N * 3
        x_src, e_src, k_e_src = params_src['x'], params_src['e'], params_src['k_e']
        
    driving_frames = []
    driven_frames = []
    src_frames = []
    driven_meshes = []
    
    # filtering noise
    MIN_CUTOFF = 1.0
    BETA = 1.0
    num_frames = len(drv_dataloader)
    fps = 25
    times = np.linspace(0, num_frames / fps, num_frames)
    
    source_mesh_dict = {}
    for k in src_data['mesh'].keys():
        source_mesh_dict[k] = torch.tensor(src_data['mesh'][k][0]).detach().cpu()
    source_mesh_dict['he_value'] = source_mesh_dict['value']
    denormalize(he_estimator, source_mesh_dict, src_data['frame'].cuda(), reference_info, is_src=True)

    
    for i, drv_data in enumerate(tqdm(drv_dataloader)):
        with torch.no_grad():
            src, drv, driven = model.module.drive(drv_data, x_src, k_e_src, x_drv=x_drv, k_e_drv=k_e_drv)
        A = np.array([-1, -1, 1 / 2]).astype('float32')[np.newaxis]
        driven_mesh = config['dataset_params']['frame_shape'][0] * (driven.detach().cpu().numpy() - A) // 2   # total_section_values x 3

        driven_mesh_shape = driven_mesh.shape
        x = driven_mesh.reshape(-1)
        
        if i == 0:
            filter_value = OneEuroFilter(times[0], x, min_cutoff=MIN_CUTOFF, beta=BETA)
        else:
            x = filter_value(times[i], x)
        
        driven_mesh = x.reshape(driven_mesh_shape)
        driven = torch.tensor((2 * driven_mesh / config['dataset_params']['frame_shape'][0]) + A).to(driven.device)
        driving_mesh = config['dataset_params']['frame_shape'][0] * (drv.detach().cpu().numpy() - A) // 2   # total_section_values x 3
        driven_frame = draw_section(driven_mesh[:, :2].astype('int32'), config['dataset_params']['frame_shape'])
        driving_frame = draw_section(driving_mesh[:, :2].astype('int32'), config['dataset_params']['frame_shape'])
        src_mesh = config['dataset_params']['frame_shape'][0] * (src.detach().cpu().numpy() - A) // 2   # total_section_values x 3
        src_frame = draw_section(src_mesh[:, :2].astype('int32'), config['dataset_params']['frame_shape'])

        source_mesh = src_data['mesh']
        driven_full_mesh = torch.tensor(source_mesh['value'][0]).detach().cpu()
        driven_full_mesh[section_indices] = driven.detach().cpu()
        
        driving_frames.append(driving_frame)
        src_frames.append(src_frame)
        driven_mesh_dict = {}
        for k in drv_data['mesh'].keys():
            driven_mesh_dict[k] = torch.tensor(drv_data['mesh'][k][0]).detach().cpu()
        driven_mesh_dict['c'] = source_mesh_dict['c']
        driven_mesh_dict['driven_sections'] = driven.detach().cpu()
        driven_mesh_dict['he_value'] = driven_full_mesh
        denormalize(he_estimator, driven_mesh_dict, drv_data['frame'].cuda(), reference_info, bias=source_mesh_dict['he_bias'], src_value=source_mesh_dict['value'])
        driven_meshes.append(driven_mesh_dict)
        driven_frame = draw_section((config['dataset_params']['frame_shape'][0] * (driven_mesh_dict['he_raw_value'].numpy() - A)[section_indices, :2] // 2).astype(np.int32), config['dataset_params']['frame_shape'])
        driven_frames.append(driven_frame)
        # print(f'mesh difference: {driven_mesh_dict["driven_sections"] - source_mesh_dict["value"][section_indices]}')
        
    src_raw_mesh = config['dataset_params']['frame_shape'][0] * (model.module.concat_section(model.module.split_section(src_data['mesh']['value'])).detach().cpu().numpy()[0] + 1) // 2 
    src_raw_frame = draw_section(src_raw_mesh[:, :2].astype('int32'), config['dataset_params']['frame_shape'])
    src_raw_frames = [src_raw_frame] * len(src_frames)
    
    imageio.mimsave(os.path.join(res_dir, 'driving.mp4'), driving_frames, fps=25)
    imageio.mimsave(os.path.join(res_dir, 'driven.mp4'), driven_frames, fps=25)
    imageio.mimsave(os.path.join(res_dir, 'source.mp4'), src_frames, fps=25)
    imageio.mimsave(os.path.join(res_dir, 'source_raw.mp4'), src_raw_frames, fps=25)
    
    torch.save(np.stack(driven_meshes, axis=0), os.path.join(res_dir, 'driven_meshes.pt'))
    torch.save(source_mesh_dict, os.path.join(res_dir, 'source_mesh.pt'))
    
if __name__ == "__main__":
    print('running')
    if sys.version_info[0] < 3:
        raise Exception("You must use Python 3 or higher. Recommended version is Python 3.7")
    os.environ['CUDA_VISIBLE_DEVICES']='1'
    parser = ArgumentParser()
    parser.add_argument("--config", default=None, help="path to config")
    parser.add_argument("--mode", default="train", choices=["train",])
    parser.add_argument("--gen", default="spade", choices=["original", "spade"])
    parser.add_argument("--log_dir", default='log_headmodel', help="path to log into")
    parser.add_argument("--checkpoint", required=True, help="path to checkpoint to restore")
    parser.add_argument("--checkpoint_posemodel", default='/home/server25/minyeong_workspace/BFv2v/ckpt/00000189-checkpoint.pth.tar', help="path to he_estimator checkpoint")
    
    parser.add_argument("--device_ids", default="0", type=lambda x: list(map(int, x.split(','))),
                        help="Names of the devices comma separated.")
    parser.add_argument("--verbose", dest="verbose", action="store_true", help="Print model architecture")
    parser.set_defaults(verbose=False)
    parser.add_argument("--coef_e_tilda", type=float, default=1.0)
    parser.add_argument("--coef_e_prime", type=float, default=1.0)
    parser.add_argument("--src_img", type=str, required=True)
    parser.add_argument("--drv_vid", type=str, required=True)
    parser.add_argument("--res_dir", type=str)
    parser.add_argument("--threshold", type=float)
    
    opt = parser.parse_args()

    if opt.config is None:
        opt.config = glob.glob(os.path.join(os.path.dirname(opt.checkpoint), '*.yaml'))[0]

    with open(opt.config) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
        
    # united single segment
    sections = config['model_params']['common_params']['headmodel_sections']
    for sec in sections:
        # print(f'seciton: {sec}')
        if len(sec[0]) == 0:
            sec[0] = list(range(478))
            
    config['train_params']['headmodel_sections'] = config['model_params']['common_params']['headmodel_sections']
    config['train_params']['coef_e_tilda'] = opt.coef_e_tilda
    config['train_params']['coef_e_prime'] = opt.coef_e_prime
    
    src_dataset = SingleImageDataset(opt.src_img, frame_shape=config['dataset_params']['frame_shape'])
    drv_dataset = SingleVideoDataset(opt.drv_vid, frame_shape=config['dataset_params']['frame_shape'])
    ref_path = 'frame_reference.png'
    reference_dataset = SingleImageDataset(ref_path)
    
    model = HeadModel(config['train_params']).cuda()
    model = DataParallelWithCallback(model)
    ckpt = torch.load(opt.checkpoint)
    model.load_state_dict(ckpt['headmodel'])
    model.module.print_statistics()
    
    he_estimator = HEEstimator(**config['model_params']['he_estimator_params'],
                               **config['model_params']['common_params'])

    if torch.cuda.is_available():
        he_estimator.to(opt.device_ids[0])

    ckpt = torch.load(opt.checkpoint_posemodel)
    he_estimator.load_state_dict(ckpt['he_estimator'])
    he_estimator.eval()
    
    res_dir = os.path.join(*os.path.split(opt.checkpoint)[:-1], 'result') if opt.res_dir is None else opt.res_dir
    if not os.path.exists(res_dir):
        os.makedirs(res_dir)
    
    print(f'Src Dataset Size: {len(src_dataset)}')
    print(f'Drv Dataset Size: {len(drv_dataset)}')    
    
    
    main(config, model, res_dir, src_dataset, drv_dataset, threshold=opt.threshold, he_estimator=he_estimator, reference_dataset=reference_dataset)

