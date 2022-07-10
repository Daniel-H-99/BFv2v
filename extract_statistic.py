import matplotlib

matplotlib.use('Agg')
import math
import os, sys
import yaml
from argparse import ArgumentParser
from time import gmtime, strftime
from shutil import copy

from frames_dataset import FramesDataset

import torch
import warnings

warnings.filterwarnings(action='ignore')

from tqdm import trange
from tqdm import tqdm
import torch

from torch.utils.data import DataLoader

from modules.headmodel import HeadModel

from logger import Logger

from frames_dataset import DatasetRepeater
from sync_batchnorm import DataParallelWithCallback

import torch.nn as nn

        
def main(config, model, log_dir, dataset, eval_dataset=None):
    train_params = config['train_params']
    
    model = DataParallelWithCallback(model)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=train_params['lr_headmodel'], betas=(0.5, 0.999))
    
    start_epoch = 0

    dataloader = DataLoader(dataset, batch_size=train_params['batch_size'], shuffle=True, num_workers=0, drop_last=True)
    if eval_dataset is not None:
        eval_dataloader = DataLoader(eval_dataset, batch_size=30, shuffle=False, num_workers=0, drop_last=False)
    else:
        eval_dataloader = None
        
    recon_weight_0 = train_params['loss_weights']['recon']
    
    min_loss = 10000
    min_epoch = 0
    
    with Logger(log_dir=log_dir, visualizer_params=config['visualizer_params'], checkpoint_freq=train_params['checkpoint_freq']) as logger:
        for epoch in trange(start_epoch, train_params['num_epochs']):
            if epoch == 0:
                train_params['loss_weights']['recon'] = 0
            else:
                train_params['loss_weights']['recon'] = recon_weight_0
                
            for x in tqdm(dataloader):
                print('on training dataset')
                losses, generated = model(x)
                generated.update({
                    'raw_source_X': {'value': generated['raw_source_X']}, \
                    'raw_driving_X': {'value': generated['raw_driving_X']}, \
                    'source_x': {'value': generated['source_x']}, \
                    'driving_x': {'value': generated['driving_x']}, \
                    'source_X': {'value': (generated['driving_x'] + generated['source_e']).detach().cpu()}, \
                    'driving_X': {'value': (generated['source_x'] + generated['driving_e']).detach().cpu()},  \
                    'source_scale': {'value': generated['source_scale'].detach().cpu()}, \
                    'driving_scale': {'value': generated['driving_scale'].detach().cpu()
                }})
                                 
                loss_values = [val.mean() for val in losses.values()]
                loss = sum(loss_values)

                loss.backward()
                optimizer.step()
                optimizer.zero_grad()

                losses = {key: value.mean().detach().data.cpu().numpy() for key, value in losses.items()}
            
                logger.log_iter(losses)
            
            if eval_dataloader is not None:
                logger.log("[Train Log]")
                logger.log_scores()
                
                eval_loss = 0
                for x in tqdm(eval_dataloader):
                    with torch.no_grad():
                        losses, generated = model(x, eval=True)
                    generated.update({
                        'raw_source_X': {'value': generated['raw_source_X']}, \
                        'raw_driving_X': {'value': generated['raw_driving_X']}, \
                        'source_x': {'value': generated['source_x']}, \
                        'driving_x': {'value': generated['driving_x']}, \
                        'source_X': {'value': (generated['driving_x'] + generated['source_e']).detach().cpu()}, \
                        'driving_X': {'value': (generated['source_x'] + generated['driving_e']).detach().cpu()},  \
                        'source_scale': {'value': generated['source_scale'].detach().cpu()}, \
                        'driving_scale': {'value': generated['driving_scale'].detach().cpu()
                    }})
                                    
                    loss_values = [val.mean() for val in losses.values()]
                    loss = losses['recon'].mean()
                    eval_loss += len(generated['raw_source_X']) * loss.detach()
                    losses = {key: value.mean().detach().data.cpu().numpy() for key, value in losses.items()}
                    logger.log_iter(losses=losses)
                
                logger.log("[Eval Log]")
                
                if eval_loss < min_loss and eval_loss > 0:
                    min_loss = eval_loss
                    logger.log("Best Epoch")
                    
            logger.log_epoch(epoch, {'headmodel': model,
                                     'optimizer_headmodel': optimizer}, inp=x, out=generated)
            
if __name__ == "__main__":
    print('running')
    if sys.version_info[0] < 3:
        raise Exception("You must use Python 3 or higher. Recommended version is Python 3.7")
    os.environ['CUDA_VISIBLE_DEVICES']='1'
    parser = ArgumentParser()
    parser.add_argument("--config", default="config/vox-256-headmodel_v5.yaml", help="path to config")
    parser.add_argument("--mode", default="train", choices=["train",])
    parser.add_argument("--gen", default="spade", choices=["original", "spade"])
    parser.add_argument("--log_dir", default='log_headmodel', help="path to log into")
    parser.add_argument("--checkpoint", default=None, help="path to checkpoint to restore")
    
    parser.add_argument("--device_ids", default="0", type=lambda x: list(map(int, x.split(','))),
                        help="Names of the devices comma separated.")
    parser.add_argument("--verbose", dest="verbose", action="store_true", help="Print model architecture")
    parser.set_defaults(verbose=False)
    parser.add_argument("--num_pc", default=None)
    parser.add_argument("--coef_e_tilda", type=float, default=1.0)
    parser.add_argument("--coef_e_prime", type=float, default=1.0)
    
    opt = parser.parse_args()
    with open(opt.config) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
        if opt.num_pc is not None:
            num_pc_per_section = opt.num_pc.split(',')
            for sec, pc in zip(config['model_params']['common_params']['headmodel_sections'], num_pc_per_section):
                sec[1] = int(pc)
            
    # united single segment
    sections = config['model_params']['common_params']['headmodel_sections']
    for sec in sections:
        # print(f'seciton: {sec}')
        if len(sec[0]) == 0:
            sec[0] = list(range(478))

    config['train_params']['headmodel_sections'] = config['model_params']['common_params']['headmodel_sections']
    config['train_params']['sections'] = config['model_params']['common_params']['sections']
    config['train_params']['coef_e_tilda'] = opt.coef_e_tilda
    config['train_params']['coef_e_prime'] = opt.coef_e_prime
    
    dataset = FramesDataset(is_train=(opt.mode == 'train'), **config['dataset_params'], train_params=config['train_params'])
    eval_dataset = FramesDataset(is_train=False, **config['dataset_params'], train_params=config['train_params'])
    model = HeadModel(config['train_params']).cuda()
    
    if opt.checkpoint is not None:
        log_dir = opt.checkpoint
    else:
        log_dir = os.path.join(opt.log_dir, os.path.basename(opt.config).split('.')[0])
        log_dir += ' {} {}'.format(opt.coef_e_tilda, opt.coef_e_prime)
        log_dir += ' ' + strftime("%d_%m_%y_%H.%M.%S", gmtime())
    
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    with open(os.path.join(log_dir, 'config.yaml'), 'w') as f:
        yaml.dump(config,f)

    print(f'Dataset Size: {len(dataset)}')
    print(f'Eval Dataset Size: {len(eval_dataset)}')    
    
    main(config, model, log_dir, dataset, eval_dataset=eval_dataset)

