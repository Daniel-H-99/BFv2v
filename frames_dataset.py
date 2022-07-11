import os
from skimage import io, img_as_float32, img_as_ubyte
from skimage.color import gray2rgb
from sklearn.model_selection import train_test_split
from imageio import mimread
import imageio

import numpy as np
from torch.utils.data import Dataset
import pandas as pd
from augmentation import AllAugmentationTransform
from mesh_augmentation import AllAugmentationWithMeshTransform
from torchvision import transforms
import random
import glob
from datetime import datetime
import torch
import cv2
from utils.util import extract_openface_mesh, get_mesh_image, draw_section, draw_mouth_mask
import warnings
warnings.filterwarnings(action='ignore')

def read_video(name, frame_shape):
    """
    Read video which can be:
      - an image of concatenated frames
      - '.mp4' and'.gif'
      - folder with videos
    """

    print(f'reading viddo {name}...')
    if os.path.isdir(name):
        print(f'{name} is direectory...')
        frames = sorted(os.listdir(name))
        num_frames = len(frames)
        video_array = np.array(
            [img_as_float32(io.imread(os.path.join(name, frames[idx]))) for idx in range(num_frames)])
    elif name.lower().endswith('.png') or name.lower().endswith('.jpg'):
        image = io.imread(name)

        if len(image.shape) == 2 or image.shape[2] == 1:
            image = gray2rgb(image)

        if image.shape[2] == 4:
            image = image[..., :3]

        image = img_as_float32(image)

        video_array = np.moveaxis(image, 1, 0)

        video_array = video_array.reshape((-1,) + frame_shape)
        video_array = np.moveaxis(video_array, 1, 2)
    elif name.lower().endswith('.gif') or name.lower().endswith('.mp4') or name.lower().endswith('.mov'):
        # video = np.array(mimread(name, memtest=False))
        reader = imageio.get_reader(name)
        video = np.array([f for f in reader])
            
        if len(video.shape) == 3:
            video = np.array([gray2rgb(frame) for frame in video])
        if video.shape[-1] == 4:
            video = video[..., :3]
        video_array = img_as_float32(video)
    else:
        raise Exception("Unknown file extensions  %s" % name)

    return video_array



class SingleImageDataset(Dataset):
    def __init__(self, image_path, frame_shape=(256, 256, 3)):
        self.image_path = image_path
        self.frame_shape = tuple(frame_shape)
        self.reference_dict = torch.load('mesh_dict_reference.pt')

    def __len__(self):
        return 1

    def __getitem__(self, idx):
        frame = cv2.resize(img_as_float32(cv2.imread(self.image_path)), self.frame_shape[:2])
        print(f'frame shape: {frame.shape}')
        L = self.frame_shape[0]
        mesh = extract_mesh(img_as_ubyte(frame), self.reference_dict) # {value (N x 3), R (3 x 3), t(3 x 1), c1}
        A = np.array([-1, -1, 1 / 2], dtype='float32')[:, np.newaxis] # 3 x 1
        mesh['value'] = np.array(mesh['value'], dtype='float32') * 2 / L  + np.squeeze(A, axis=-1)[None]
        mesh['R'] = np.array(mesh['R'], dtype='float32')
        mesh['c'] = np.array(mesh['c'], dtype='float32')
        t = np.array(mesh['t'], dtype='float32')
        mesh['t'] = (np.eye(3).astype(np.float32) - mesh['c'] * mesh['R']) @ A + t * 2 / L
        mesh['raw_value'] = np.array(mesh['raw_value'], dtype='float32') * 2 / L + np.squeeze(A, axis=-1)[None]
       
        out = {}
        # if self.is_train:
        frame = np.array(frame, dtype='float32')
        out['frame'] = frame.transpose((2, 0, 1))
        out['mesh'] = mesh
        # else:
        #     video = np.array(video_array, dtype='float32')
        #     out['video'] = video.transpose((3, 0, 1, 2))
        #     out['mesh'] = meshes
        return out

class SingleVideoDataset(Dataset):
    def __init__(self, video_path, frame_shape=(256, 256, 3)):
        self.video_path = video_path
        self.frame_shape = tuple(frame_shape)
        self.reference_dict = torch.load('mesh_dict_reference.pt')
        self.frames = read_video(self.video_path, frame_shape) # L x H x W x 3
        
    def __len__(self):
        return len(self.frames)

    def __getitem__(self, idx):
        while True:
            try:
                frame = self.frames[idx]
                frame = cv2.resize(frame, self.frame_shape[:2])
                L = self.frame_shape[0]
                mesh = extract_mesh(img_as_ubyte(frame), self.reference_dict) # {value (N x 3), R (3 x 3), t(3 x 1), c1}
                A = np.array([-1, -1, 1 / 2], dtype='float32')[:, np.newaxis] # 3 x 1
                mesh['value'] = np.array(mesh['value'], dtype='float32') * 2 / L  + np.squeeze(A, axis=-1)[None]
                mesh['R'] = np.array(mesh['R'], dtype='float32')
                mesh['c'] = np.array(mesh['c'], dtype='float32')
                t = np.array(mesh['t'], dtype='float32')
                mesh['t'] = (np.eye(3).astype(np.float32) - mesh['c'] * mesh['R']) @ A + t * 2 / L
                mesh['raw_value'] = np.array(mesh['raw_value'], dtype='float32') * 2 / L + np.squeeze(A, axis=-1)[None]

                out = {}
                # if self.is_train:
                frame = np.array(frame, dtype='float32')
                out['frame'] = frame.transpose((2, 0, 1))
                out['mesh'] = mesh
                # else:
                #     video = np.array(video_array, dtype='float32')
                #     out['video'] = video.transpose((3, 0, 1, 2))
                #     out['mesh'] = meshes
                break
            except Exception as e:
                print(f'error occured for idx: {idx} -> {e}')
                idx += 1
        return out


class FramesDataset3(Dataset):
    """
    Dataset of videos, each video can be represented as:
      - an image of concatenated frames
      - '.mp4' or '.gif'
      - folder with all frames
    """

    def __init__(self, root_dir, frame_shape=(256, 256, 3), id_sampling=False, is_train=True,
                 random_seed=0, pairs_list=None, augmentation_params=None, train_params=None):
        self.train_params = train_params
        self.sections = self.train_params['sections']
        self.root_dir = root_dir
        self.videos = os.listdir(root_dir)
        self.frame_shape = tuple(frame_shape)
        self.pairs_list = pairs_list
        self.id_sampling = id_sampling
        self.reference_dict = torch.load('mesh_dict_reference.pt')
        if os.path.exists(os.path.join(root_dir, 'train')):
            # assert os.path.exists(os.path.join(root_dir, 'test'))
            print("Use predefined train-test split.")
            if id_sampling:
                train_videos = {os.path.basename(video).split('#')[0] for video in
                                os.listdir(os.path.join(root_dir, 'train'))}
                train_videos = list(train_videos)
            else:
                train_videos = os.listdir(os.path.join(root_dir, 'train'))
            # test_videos = os.listdir(os.path.join(root_dir, 'test'))
            self.root_dir = os.path.join(self.root_dir, 'train' if is_train else 'test')
        else:
            print("Use random train-test split.")
            train_videos, test_videos = train_test_split(self.videos, random_state=random_seed, test_size=0.2)

        if is_train:
            self.videos = train_videos
        else:
            self.videos = test_videos

        self.is_train = is_train

        if self.is_train:
            self.transform = AllAugmentationTransform(**augmentation_params)
        else:
            self.transform = None
            
    def __len__(self):
        return len(self.videos) // 100

    def split_section(self, X):
        res = []
        for i, sec in enumerate(self.sections):
            res.append(X[sec[0]])
        return res

    
    def concat_section(self, sections):
        # sections[]: (num_sections) x -1 x 3
        return np.concatenate(sections, axis=0)

    def get_mouth_image(self, mesh):
        mouth = draw_mouth_mask(mesh[:, :2].astype(np.int32), self.frame_shape)
        mouth = mouth[:, :, :1].astype(np.float32).transpose((2, 0, 1))
        
        return mouth
    
    def get_mesh_image_section(self, mesh):
        # mesh: N0 x 3
        # print(f'mesh type: {mesh.type()}')
        # mouth_mask = (255 * draw_mouth_mask(mesh[:, :2].numpy().astype(np.int32), self.frame_shape)).astype(np.int32)
        # print(f'mouth mask shape {mouth_mask.type()}')
        sections = self.concat_section(self.split_section(mesh))
        # print(f'sections shape: {sections.shape}')
        
        secs = draw_section(sections[:, :2].astype(np.int32), self.frame_shape, split=False) # (num_sections) x H x W x 3
        # print(f'draw section done')
        secs = secs[:, :, :1].astype(np.float32).transpose((2, 0, 1)) / 255.0
        # secs = [sec[:, :, :1].astype(np.float32).transpose((2, 0, 1)) / 255.0 for sec in secs]
        # print('got mesh image sections')
        return secs
    
    def __getitem__(self, idx):
        while True:
            try:
                idx = (idx + int(datetime.now().timestamp())) % (len(self.videos))
                if self.is_train and self.id_sampling:
                    name = self.videos[idx]
                    path = np.random.choice(glob.glob(os.path.join(self.root_dir, name)))
                else:
                    name = self.videos[idx]
                    path = os.path.join(self.root_dir, name)

                video_name = os.path.basename(path)

                # if self.is_train and os.path.isdir(path):
                frames = os.listdir(path)
                num_frames = len(frames)
                frame_idx = np.sort(np.random.choice(num_frames, replace=False, size=2))
                frame_idx[1] = (frame_idx[1] + 100) % num_frames
                raw_video_array = [io.imread(os.path.join(path, frames[(idx + int(datetime.now().timestamp())) % num_frames])) for idx in frame_idx]
                video_array = np.stack([cv2.resize(img_as_float32(frame), self.frame_shape[:2]) for frame in raw_video_array], axis=0)
                transform_hopenet =  transforms.Compose([
                                                        transforms.ToTensor(),
                                                        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
                hopenet_video_array = [transform_hopenet(cv2.resize(frame, (224, 224))) for frame in raw_video_array]
                
                # else:
                #     raw_video_array = read_video(path, frame_shape=self.frame_shape)
                #     num_frames = len(raw_video_array)
                #     frame_idx = np.sort(np.random.choice(num_frames, replace=True, size=2)) if self.is_train else range(
                #         num_frames)
                #     frames_idx =[((fid + int(datetime.now().timestamp())) % num_frames) for fid in frame_idx]
                #     frames_idx[1] = (frames_idx[1] + 100) % num_frames
                #     video_array = np.stack([cv2.resize(img_as_float32(raw_video_array[fid]), self.frame_shape[:2]) for fid in frame_idx], axis=0)
                
                
                #     transform_hopenet =  transforms.Compose([
                #                                             transforms.ToTensor(),
                #                                             transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
                #     hopenet_video_array = [transform_hopenet(cv2.resize(raw_video_array[fid], (224, 224))) for fid in frame_idx]
                    
                if self.transform is not None:
                    video_array = self.transform(video_array)
                    
                meshes = []
                for frame in video_array:
                    L = self.frame_shape[0]
                    mesh = extract_openface_mesh(img_as_ubyte(frame)) # {value (N x 3), R (3 x 3), t(3 x 1), c1}
                    A = np.array([[-1, -1, 0]], dtype='float32') # 3 x 1
                    # # print(f'value: {mesh["value"]}')
                    # mesh['value'] = np.array(mesh['value'], dtype='float32') * 2 / L  + np.squeeze(A, axis=-1)[None]
                    # mesh['R'] = np.array(mesh['R'], dtype='float32')
                    # mesh['c'] = np.array(mesh['c'], dtype='float32')
                    # t = np.array(mesh['t'], dtype='float32')
                    # mesh['t'] = (np.eye(3).astype(np.float32) - mesh['c'] * mesh['R']) @ A + t * 2 / L
                    # # print('checkpoint 1')

                    # mesh['mesh_img'] = (get_mesh_image(mesh['raw_value'], self.frame_shape)[:, :, [0]] / 255.0).transpose((2, 0, 1))
                    mesh['mesh_img_sec'] =  self.get_mesh_image_section(mesh['raw_value'])
                    # print('msh img sec got')
                    
                    # mesh['mouth_img'] = self.get_mouth_image(mesh['raw_value'].numpy())
                    # print(f'mouth image shape: {mesh["mouth_img"].shape}')
                    mesh['raw_value'] = np.array(mesh['raw_value'], dtype='float32') * 2 / L + A
                    # print('raw value got')
                    # print('checkpoint 2')
                    # print(f'data type: {mesh["value"].dtype}')
                    meshes.append(mesh)
                            
                        # ### Make intermediate target mesh ###
                        # src_mesh = meshes[0]
                        # drv_mesh = meshes[1]
                        # target_mesh = (1 / src_mesh['c'][np.newaxis, np.newaxis]) * np.einsum('ij,nj->ni', np.linalg.inv(src_mesh['R']), drv_mesh['value'] - src_mesh['t'][np.newaxis, :, 0])
                        # drv_mesh['intermediate_value'] = target_mesh
                        # target_mesh = L * (target_mesh - np.squeeze(A, axis=-1)[None]) // 2
                        # drv_mesh['intermediate_mesh_img_sec'] = self.get_mesh_image_section(target_mesh)
                        
                break
            
            except Exception as e:
                print(f'error: {e}')
                continue
            
            
        out = {}
        # if self.is_train:
        source = np.array(video_array[0], dtype='float32')
        driving = np.array(video_array[1], dtype='float32')
        out['driving'] = driving.transpose((2, 0, 1))
        out['source'] = source.transpose((2, 0, 1))
        out['driving_mesh'] = meshes[1]
        out['source_mesh'] = meshes[0]
        # out['hopenet_source'] = hopenet_video_array[0]
        # out['hopenet_driving'] = hopenet_video_array[1]
        
        
        # else:
        #     video = np.array(video_array, dtype='float32')
        #     out['video'] = video.transpose((3, 0, 1, 2))
        #     out['mesh'] = meshes
            
        out['name'] = video_name

        return out
    
class FramesDataset2(Dataset):
    """
    Dataset of videos, each video can be represented as:
      - an image of concatenated frames
      - '.mp4' or '.gif'
      - folder with all frames
    """

    def __init__(self, root_dir, frame_shape=(256, 256, 3), is_train=True,
                 random_seed=0, augmentation_params=None, train_params=None):
        self.train_params = train_params
        self.sections = self.train_params['sections']
        self.root_dir = root_dir
        self.videos = os.listdir(root_dir)
        self.frame_shape = tuple(frame_shape)
        self.reference_dict = torch.load('mesh_dict_reference.pt')

        if os.path.exists(os.path.join(root_dir, 'train')):
            assert os.path.exists(os.path.join(root_dir, 'test'))
            print("Use predefined train-test split.")
            train_ids = os.listdir(os.path.join(root_dir, 'train'))
            test_ids = os.listdir(os.path.join(root_dir, 'test'))
            self.root_dir = os.path.join(self.root_dir, 'train' if is_train else 'test')
        else:
            print("Use random train-test split.")
            train_ids, test_ids = train_test_split(self.videos, random_state=random_seed, test_size=0.2)

        if is_train:
            self.ids = train_ids
        else:
            self.ids = test_ids
            
        self.is_train = is_train

        if self.is_train:
            self.transform = AllAugmentationTransform(**augmentation_params)
        else:
            self.transform = None

    def __len__(self):
        return len(self.ids) // 100

    def split_section(self, X):
        res = []
        for i, sec in enumerate(self.sections):
            res.append(X[sec[0]])
        return res

    
    def concat_section(self, sections):
        # sections[]: (num_sections) x -1 x 3
        return np.concatenate(sections, axis=0)

    def get_mouth_image(self, mesh):
        mouth = draw_mouth_mask(mesh[:, :2].astype(np.int32), self.frame_shape)
        mouth = mouth[:, :, :1].astype(np.float32).transpose((2, 0, 1))
        
        return mouth
    
    def get_mesh_image_section(self, mesh):
        # mesh: N0 x 3
        sections = self.concat_section(self.split_section(mesh))
        # print(f'sections shape: {sections.shape}')
        secs = draw_section(sections[:, :2].astype(np.int32), self.frame_shape, split=False) # (num_sections) x H x W x 3
        # print(f'draw section done')
        secs = sec[:, :, :1].astype(np.float32).transpose((2, 0, 1)) / 255.0
        # secs = [sec[:, :, :1].astype(np.float32).transpose((2, 0, 1)) / 255.0 for sec in secs]
        # print('got mesh image sections')
        return secs
            
            
    def __getitem__(self, idx):
        while True:
            try:
                magic_num = int(datetime.now().timestamp())
                idx = (idx + magic_num) % (len(self.videos))
                id = self.ids[idx]
                chunk = np.random.choice(os.listdir((os.path.join(self.root_dir, id))))
                path = np.random.choice(glob.glob(os.path.join(self.root_dir, id, chunk, '*.mp4')))

                video_name = os.path.basename(path)

                video_array = read_video(path, frame_shape=self.frame_shape)
                num_frames = len(video_array)
                frame_idx = np.sort(np.random.choice(num_frames, replace=False, size=2)) if self.is_train else range(
                    num_frames)
                frame_idx[1] = (frame_idx[1] + magic_num) % num_frames
                
                raw_video_array = [video_array[(idx + magic_num) % num_frames] for idx in frame_idx]
                video_array = np.stack([cv2.resize(frame, self.frame_shape[:2]) for frame in raw_video_array], axis=0)

                if self.transform is not None:
                    video_array = self.transform(video_array)

                meshes = []
                for frame in video_array:
                    L = self.frame_shape[0]
                    mesh = extract_mesh(img_as_ubyte(frame), self.reference_dict) # {value (N x 3), R (3 x 3), t(3 x 1), c1}
                    A = np.array([-1, -1, 1 / 2], dtype='float32')[:, np.newaxis] # 3 x 1
                    # print(f'value: {mesh["value"]}')
                    mesh['value'] = np.array(mesh['value'], dtype='float32') * 2 / L  + np.squeeze(A, axis=-1)[None]
                    mesh['R'] = np.array(mesh['R'], dtype='float32')
                    mesh['c'] = np.array(mesh['c'], dtype='float32')
                    t = np.array(mesh['t'], dtype='float32')
                    mesh['t'] = (np.eye(3).astype(np.float32) - mesh['c'] * mesh['R']) @ A + t * 2 / L
                    # print('checkpoint 1')

                    mesh['mesh_img'] = (get_mesh_image(mesh['raw_value'], self.frame_shape)[:, :, [0]] / 255.0).transpose((2, 0, 1))
                    mesh['mesh_img_sec'] =  self.get_mesh_image_section(mesh['raw_value'])

                    # print('msh img sec got')
                    
                    # mesh['mouth_img'] = self.get_mouth_image(mesh['raw_value'].numpy())
                    # print(f'mouth image shape: {mesh["mouth_img"].shape}')
                    mesh['raw_value'] = np.array(mesh['raw_value'], dtype='float32') * 2 / L + np.squeeze(A, axis=-1)[None]
                    # print('raw value got')
                    # print('checkpoint 2')
                    meshes.append(mesh)
                    
                # ### Make intermediate target mesh ###
                # src_mesh = meshes[0]
                # drv_mesh = meshes[1]
                # target_mesh = (1 / src_mesh['c'][np.newaxis, np.newaxis]) * np.einsum('ij,nj->ni', np.linalg.inv(src_mesh['R']), drv_mesh['value'] - src_mesh['t'][np.newaxis, :, 0])
                # drv_mesh['intermediate_value'] = target_mesh
                # target_mesh = L * (target_mesh - np.squeeze(A, axis=-1)[None]) // 2
                # drv_mesh['intermediate_mesh_img_sec'] = self.get_mesh_image_section(target_mesh)
                
                break
            
            except Exception as e:
                print(f'error: {e}')
                continue
            
            
        out = {}
        # if self.is_train:
        source = np.array(video_array[0], dtype='float32')
        driving = np.array(video_array[1], dtype='float32')
        out['driving'] = driving.transpose((2, 0, 1))
        out['source'] = source.transpose((2, 0, 1))
        out['driving_mesh'] = meshes[1]
        out['source_mesh'] = meshes[0]
        # out['hopenet_source'] = hopenet_video_array[0]
        # out['hopenet_driving'] = hopenet_video_array[1]
        
        
        # else:
        #     video = np.array(video_array, dtype='float32')
        #     out['video'] = video.transpose((3, 0, 1, 2))
        #     out['mesh'] = meshes
            
        out['name'] = video_name

        return out


class FramesDataset(Dataset):
    """
    Dataset of videos, each video can be represented as:
      - an image of concatenated frames
      - '.mp4' or '.gif'
      - folder with all frames
    """

    def __init__(self, root_dir, frame_shape=(256, 256, 3), id_sampling=False, is_train=True,
                 random_seed=0, pairs_list=None, augmentation_params=None, train_params=None):
        self.train_params = train_params
        self.sections = self.train_params['sections']
        self.root_dir = root_dir
        self.videos = os.listdir(root_dir)
        self.frame_shape = tuple(frame_shape)
        self.pairs_list = pairs_list
        self.id_sampling = id_sampling
        self.reference_dict = torch.load('mesh_dict_reference.pt')
        if os.path.exists(os.path.join(root_dir, 'train')):
            # assert os.path.exists(os.path.join(root_dir, 'test'))
            print("Use predefined train-test split.")
            if id_sampling:
                train_videos = {os.path.basename(video).split('#')[0] for video in
                                os.listdir(os.path.join(root_dir, 'train'))}
                train_videos = list(train_videos)
            else:
                train_videos = os.listdir(os.path.join(root_dir, 'train'))
            # test_videos = os.listdir(os.path.join(root_dir, 'test'))
            self.root_dir = os.path.join(self.root_dir, 'train' if is_train else 'test')
        else:
            print("Use random train-test split.")
            train_videos, test_videos = train_test_split(self.videos, random_state=random_seed, test_size=0.2)

        if is_train:
            self.videos = train_videos
        else:
            self.videos = test_videos

        self.is_train = is_train

        if self.is_train:
            self.transform = AllAugmentationTransform(**augmentation_params)
        else:
            self.transform = None
            
    def __len__(self):
        return len(self.videos) // 100

    def split_section(self, X):
        res = []
        for i, sec in enumerate(self.sections):
            res.append(X[sec[0]])
        return res

    
    def concat_section(self, sections):
        # sections[]: (num_sections) x -1 x 3
        return np.concatenate(sections, axis=0)

    def get_mouth_image(self, mesh):
        mouth = draw_mouth_mask(mesh[:, :2].astype(np.int32), self.frame_shape)
        mouth = mouth[:, :, :1].astype(np.float32).transpose((2, 0, 1))
        
        return mouth
    
    def get_mesh_image_section(self, mesh):
        # mesh: N0 x 3
        # print(f'mesh type: {mesh.type()}')
        # mouth_mask = (255 * draw_mouth_mask(mesh[:, :2].numpy().astype(np.int32), self.frame_shape)).astype(np.int32)
        # print(f'mouth mask shape {mouth_mask.type()}')
        sections = self.concat_section(self.split_section(mesh))
        # print(f'sections shape: {sections.shape}')
        
        secs = draw_section(sections[:, :2].astype(np.int32), self.frame_shape, split=False) # (num_sections) x H x W x 3
        # print(f'draw section done')
        secs = secs[:, :, :1].astype(np.float32).transpose((2, 0, 1)) / 255.0
        # secs = [sec[:, :, :1].astype(np.float32).transpose((2, 0, 1)) / 255.0 for sec in secs]
        # print('got mesh image sections')
        return secs
            
    def __getitem__(self, idx):
        while True:
            try:
                idx = (idx + int(datetime.now().timestamp())) % (len(self.videos))
                if self.is_train and self.id_sampling:
                    name = self.videos[idx]
                    path = np.random.choice(glob.glob(os.path.join(self.root_dir, name)))
                else:
                    name = self.videos[idx]
                    path = os.path.join(self.root_dir, name)

                video_name = os.path.basename(path)

                # if self.is_train and os.path.isdir(path):
                frames = os.listdir(path)
                num_frames = len(frames)
                frame_idx = np.sort(np.random.choice(num_frames, replace=False, size=2))
                frame_idx[1] = (frame_idx[1] + 100) % num_frames
                raw_video_array = [io.imread(os.path.join(path, frames[(idx + int(datetime.now().timestamp())) % num_frames])) for idx in frame_idx]
                video_array = np.stack([cv2.resize(img_as_float32(frame), self.frame_shape[:2]) for frame in raw_video_array], axis=0)
                transform_hopenet =  transforms.Compose([
                                                        transforms.ToTensor(),
                                                        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
                hopenet_video_array = [transform_hopenet(cv2.resize(frame, (224, 224))) for frame in raw_video_array]
                
                # else:
                #     raw_video_array = read_video(path, frame_shape=self.frame_shape)
                #     num_frames = len(raw_video_array)
                #     frame_idx = np.sort(np.random.choice(num_frames, replace=True, size=2)) if self.is_train else range(
                #         num_frames)
                #     frames_idx =[((fid + int(datetime.now().timestamp())) % num_frames) for fid in frame_idx]
                #     frames_idx[1] = (frames_idx[1] + 100) % num_frames
                #     video_array = np.stack([cv2.resize(img_as_float32(raw_video_array[fid]), self.frame_shape[:2]) for fid in frame_idx], axis=0)
                
                
                #     transform_hopenet =  transforms.Compose([
                #                                             transforms.ToTensor(),
                #                                             transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
                #     hopenet_video_array = [transform_hopenet(cv2.resize(raw_video_array[fid], (224, 224))) for fid in frame_idx]
                    
                if self.transform is not None:
                    video_array = self.transform(video_array)

                meshes = []
                for frame in video_array:
                    L = self.frame_shape[0]
                    mesh = extract_mesh(img_as_ubyte(frame), self.reference_dict) # {value (N x 3), R (3 x 3), t(3 x 1), c1}
                    A = np.array([-1, -1, 1 / 2], dtype='float32')[:, np.newaxis] # 3 x 1
                    # print(f'value: {mesh["value"]}')
                    mesh['value'] = np.array(mesh['value'], dtype='float32') * 2 / L  + np.squeeze(A, axis=-1)[None]
                    mesh['R'] = np.array(mesh['R'], dtype='float32')
                    mesh['c'] = np.array(mesh['c'], dtype='float32')
                    t = np.array(mesh['t'], dtype='float32')
                    mesh['t'] = (np.eye(3).astype(np.float32) - mesh['c'] * mesh['R']) @ A + t * 2 / L
                    # print('checkpoint 1')

                    mesh['mesh_img'] = (get_mesh_image(mesh['raw_value'], self.frame_shape)[:, :, [0]] / 255.0).transpose((2, 0, 1))
                    mesh['mesh_img_sec'] =  self.get_mesh_image_section(mesh['raw_value'])

                    # print('msh img sec got')
                    
                    # mesh['mouth_img'] = self.get_mouth_image(mesh['raw_value'].numpy())
                    # print(f'mouth image shape: {mesh["mouth_img"].shape}')
                    mesh['raw_value'] = np.array(mesh['raw_value'], dtype='float32') * 2 / L + np.squeeze(A, axis=-1)[None]
                    # print('raw value got')
                    # print('checkpoint 2')
                    meshes.append(mesh)
                    
                ### Make intermediate target mesh ###
                src_mesh = meshes[0]
                drv_mesh = meshes[1]
                target_mesh = (1 / drv_mesh['c'][np.newaxis, np.newaxis]) * np.einsum('ij,nj->ni', np.linalg.inv(drv_mesh['R']), src_mesh['value'] - drv_mesh['t'][np.newaxis, :, 0])
                target_mesh = L * (target_mesh - np.squeeze(A, axis=-1)[None]) // 2
                drv_mesh['fake_mesh_img'] = (get_mesh_image(target_mesh, self.frame_shape)[:, :, [0]] / 255.0).transpose((2, 0, 1))

                break
            except Exception as e:
                print(f'error: {e}')
                continue
            
            
        
        out = {}
        # if self.is_train:
        source = np.array(video_array[0], dtype='float32')
        driving = np.array(video_array[1], dtype='float32')
        out['driving'] = driving.transpose((2, 0, 1))
        out['source'] = source.transpose((2, 0, 1))
        out['driving_mesh'] = meshes[1]
        out['source_mesh'] = meshes[0]
        out['hopenet_source'] = hopenet_video_array[0]
        out['hopenet_driving'] = hopenet_video_array[1]
        # else:
        #     video = np.array(video_array, dtype='float32')
        #     out['video'] = video.transpose((3, 0, 1, 2))
        #     out['mesh'] = meshes
            
        out['name'] = video_name

        return out


class DatasetRepeater(Dataset):
    """
    Pass several times over the same dataset for better i/o performance
    """

    def __init__(self, dataset, num_repeats=100):
        self.dataset = dataset
        self.num_repeats = num_repeats

    def __len__(self):
        return self.num_repeats * self.dataset.__len__()

    def __getitem__(self, idx):
        return self.dataset[idx % self.dataset.__len__()]
