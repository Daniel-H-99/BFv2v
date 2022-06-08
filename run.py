import matplotlib

matplotlib.use('Agg')

import os, sys
import yaml
from argparse import ArgumentParser
from time import gmtime, strftime
from shutil import copy

from frames_dataset import FramesDataset, SingleImageDataset

from modules.generator import OcclusionAwareGenerator, OcclusionAwareSPADEGenerator
from modules.discriminator import MultiScaleDiscriminator
from modules.keypoint_detector import KPDetector, HEEstimator
from modules.headmodel import HeadModel
from sync_batchnorm import DataParallelWithCallback

import torch

from train import train
import warnings

warnings.filterwarnings(action='ignore')

if __name__ == "__main__":
    
    if sys.version_info[0] < 3:
        raise Exception("You must use Python 3 or higher. Recommended version is Python 3.7")
<<<<<<< HEAD
    os.environ['CUDA_VISIBLE_DEVICES']='1,2'
=======
    os.environ['CUDA_VISIBLE_DEVICES']='0,1,2,3'
>>>>>>> a3f842d9c540e094d41883e5420257752a521610
    parser = ArgumentParser()
    parser.add_argument("--config", default="config/vox-256.yaml", help="path to config")
    parser.add_argument("--mode", default="train", choices=["train",])
    parser.add_argument("--gen", default="spade", choices=["original", "spade"])
    parser.add_argument("--log_dir", default='log', help="path to log into")
    parser.add_argument("--checkpoint", default=None, help="path to checkpoint to restore")
    parser.add_argument("--checkpoint_headmodel", default=None, help="path to checkpoint to restore")
    parser.add_argument("--checkpoint_posemodel", default='/home/server25/minyeong_workspace/fv2v/ckpt/00000189-checkpoint.pth.tar', help="path to he_estimator checkpoint")
    
    parser.add_argument("--device_ids", default="0,1,2,3", type=lambda x: list(map(int, x.split(','))),
                        help="Names of the devices comma separated.")
    parser.add_argument("--verbose", dest="verbose", action="store_true", help="Print model architecture")
    parser.set_defaults(verbose=False)

    opt = parser.parse_args()
    with open(opt.config) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    if opt.mode == 'train':
        config['train_params']['num_kp'] = config['model_params']['common_params']['num_kp']
        config['train_params']['sections'] = config['model_params']['common_params']['sections']
        config['train_params']['headmodel_sections'] = config['model_params']['common_params']['headmodel_sections']
        
    if opt.checkpoint is not None:
        log_dir = os.path.join(*os.path.split(opt.checkpoint)[:-1])
    else:
        log_dir = os.path.join(opt.log_dir, os.path.basename(opt.config).split('.')[0])
        log_dir += ' ' + strftime("%d_%m_%y_%H.%M.%S", gmtime())
        
    

    headmodel = HeadModel(config['train_params']).cuda()
    headmodel = DataParallelWithCallback(headmodel)
    ckpt = torch.load(opt.checkpoint_headmodel)
    headmodel.load_state_dict(ckpt['headmodel'])
    statistics = headmodel.module.export_statistics()
    del headmodel
    
    
    
    # print('pass 1')
    if opt.gen == 'original':
        generator = OcclusionAwareGenerator(**config['model_params']['generator_params'],
                                            **config['model_params']['common_params'])
    elif opt.gen == 'spade':
        generator = OcclusionAwareSPADEGenerator(**config['model_params']['generator_params'],
                                                 **config['model_params']['common_params'], headmodel=statistics)

    if torch.cuda.is_available():
        print('cuda is available')
        generator.to(opt.device_ids[0])
    if opt.verbose:
        print(generator)

    if config['train_params']['loss_weights']['generator_gan'] > 0:
        discriminator = MultiScaleDiscriminator(**config['model_params']['discriminator_params'],
                                                **config['model_params']['common_params'])
        if torch.cuda.is_available():
            discriminator.to(opt.device_ids[0])
        if opt.verbose:
            print(discriminator)
    else:
        discriminator = None

        
    
    # kp_detector = KPDetector(**config['model_params']['kp_detector_params'],
    #                          **config['model_params']['common_params'])

    # if torch.cuda.is_available():
    #     kp_detector.to(opt.device_ids[0])

    # if opt.verbose:
    #     print(kp_detector)

    # he_estimator = HEEstimator(**config['model_params']['he_estimator_params'],
    #                            **config['model_params']['common_params'])

    # if torch.cuda.is_available():
    #     he_estimator.to(opt.device_ids[0])

    # ckpt = torch.load(opt.checkpoint_posemodel)
    # he_estimator.load_state_dict(ckpt['he_estimator'])
    # he_estimator.eval()
    
    
    dataset = FramesDataset(is_train=(opt.mode == 'train'), **config['dataset_params'], train_params=config['train_params'])
    
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    if not os.path.exists(os.path.join(log_dir, os.path.basename(opt.config))):
        copy(opt.config, log_dir)

    if opt.mode == 'train':
        print("Training...")
        train(config, generator, discriminator, None, None, opt.checkpoint, log_dir, dataset, opt.device_ids)
