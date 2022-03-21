import os
from skimage import io, img_as_float32, img_as_ubyte
from skimage.color import gray2rgb
from sklearn.model_selection import train_test_split
from imageio import mimread

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
from utils.util import extract_mesh

def read_video(name, frame_shape):
    """
    Read video which can be:
      - an image of concatenated frames
      - '.mp4' and'.gif'
      - folder with videos
    """

    if os.path.isdir(name):
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
        video = np.array(mimread(name, memtest=False))
        if len(video.shape) == 3:
            video = np.array([gray2rgb(frame) for frame in video])
        if video.shape[-1] == 4:
            video = video[..., :3]
        video_array = img_as_float32(video)
    else:
        raise Exception("Unknown file extensions  %s" % name)

    return video_array

class MeshFramesDataset(Dataset):
    """
    Dataset of videos, each video can be represented as:
      - an image of concatenated frames
      - '.mp4' or '.gif'
      - folder with all frames
    """

    def __init__(self, root_dir, frame_shape=(256, 256, 3), is_train=True,
                 random_seed=0, num_prevs=0, id_sampling=False, pairs_list=None, augmentation_params=None, num_dummy_set=0):
        self.root_dir = root_dir
        self.frame_shape = tuple(frame_shape)
        self.num_prevs = num_prevs
        self.pairs_list = pairs_list
        self.id_sampling = id_sampling
        self.num_dummy_set = num_dummy_set
        # if os.path.exists(os.path.join(root_dir, 'train')):
        #     assert os.path.exists(os.path.join(root_dir, 'test'))
        #     print("Use predefined train-test split.")
        #     train_videos = os.listdir(os.path.join(root_dir, 'train'))
        #     test_videos = os.listdir(os.path.join(root_dir, 'test'))
        #     self.root_dir = os.path.join(self.root_dir, 'train' if is_train else 'test')
        # else:
        #     print("Use random train-test split.")
        #     train_videos, test_videos = train_test_split(self.videos, random_state=random_seed, test_size=0.2)

        self.videos = list(filter(lambda x: x.endswith('.mp4'), os.listdir(root_dir)))
        self.is_train = is_train

        if self.is_train:
            self.transform = AllAugmentationWithMeshTransform(**augmentation_params)
        else:
            self.transform = None
        
        self.length = {}
        print('Dataset size: {}'.format(self.__len__()))


    def __len__(self):
        length = 0
        for vid in self.videos:
            path = os.path.join(self.root_dir, vid)
            num_frames = len(os.listdir(os.path.join(path, 'img')))
            length += num_frames
            self.length[vid] = num_frames
        return length

    def __getitem__(self, idx):
        name = random.choice(self.videos)
        idx %= self.length[name]
        path = os.path.join(self.root_dir, name)

        video_name = os.path.basename(path)
    
        frames = sorted(os.listdir(os.path.join(path, 'img')))
        num_frames = len(frames)
        frame_idx = [(idx + int(datetime.now().timestamp())) % self.length[name], idx] if self.is_train else range(min(500, num_frames))

        mesh_dicts = [torch.load(os.path.join(path, 'mesh_dict', frames[frame_idx[i]].replace('.png', '.pt'))) for i in range(len(frame_idx))]
        mesh_dicts_normed = [torch.load(os.path.join(path, 'mesh_dict_normalized', frames[frame_idx[i]].replace('.png', '.pt'))) for i in range(len(frame_idx))]
        R_array = [np.array(mesh_dict['R']) for mesh_dict in mesh_dicts]
        t_array = [np.array(mesh_dict['t']) for mesh_dict in mesh_dicts]
        c_array = [np.array(mesh_dict['c']) for mesh_dict in mesh_dicts]
        mesh_array = [np.array(list(mesh_dict.values())[:478]) for mesh_dict in mesh_dicts]
        normed_mesh_array = [np.array(list(mesh_dict_normed.values())[:478]) for mesh_dict_normed in mesh_dicts_normed]
        z_array = [torch.load(os.path.join(path, 'z', frames[frame_idx[i]].replace('.png', '.pt'))) for i in range(len(frame_idx))]
        normed_z_array = [torch.load(os.path.join(path, 'z_normalized', frames[frame_idx[i]].replace('.png', '.pt'))) for i in range(len(frame_idx))]
        video_array = [img_as_float32(io.imread(os.path.join(path, 'img', frames[frame_idx[i]]))) for i in range(len(frame_idx))]
        mesh_img_array = [img_as_float32(io.imread(os.path.join(path, 'mesh_image', frames[frame_idx[i]]))) for i in range(len(frame_idx))]
        
        video_array.extend([img_as_float32(io.imread(os.path.join(path, 'img', frames[max(0, frame_idx[1] - i - 1)]))) for i in range(self.num_prevs)])
        mesh_img_array.extend([img_as_float32(io.imread(os.path.join(path, 'mesh_image', frames[max(0, frame_idx[1] - i - 1)]))) for i in range(self.num_prevs)])
        
        R_array.append(R_array[1])
        t_array.append(t_array[1])
        c_array.append(c_array[1])
        mesh_array.append(mesh_array[1])
        normed_mesh_array.append(normed_mesh_array[1])
        video_array.append(video_array[1])
        mesh_img_array.append(mesh_img_array[1])
        z_array.append(z_array[1])
        normed_z_array.append(normed_z_array[1])
        
        if self.transform is not None:
            video_array, mesh_array, R_array, t_array, c_array, mesh_img_array = self.transform(video_array, mesh_array, R_array, t_array, c_array, mesh_img_array)

        video_array = np.array(video_array, dtype='float32')
        if self.is_train:
            mesh_img_array = np.array(mesh_img_array, dtype='float32')
            normed_z_array = torch.stack(normed_z_array, dim=0).float() / 128 - 1
        mesh_array = np.array(mesh_array, dtype='float32') / 128 - 1
        normed_mesh_array = np.array(normed_mesh_array, dtype='float32') / 128 - 1
        R_array = np.array(R_array, dtype='float32')
        c_array = np.array(c_array, dtype='float32') * 128
        t_array = np.array(t_array, dtype='float32')
        t_array = t_array + np.matmul(R_array, (c_array[:, None, None] * np.ones_like(t_array)))
        z_array = torch.stack(z_array, dim=0).float() / 128 - 1

        out = {}

        source = video_array[0]
        real = video_array[1]
        driving = video_array[2]
        source_mesh = mesh_array[0]
        real_mesh = mesh_array[1]
        driving_mesh = mesh_array[2]
        source_normed_mesh = normed_mesh_array[0]
        real_normed_mesh = normed_mesh_array[1]
        driving_normed_mesh = normed_mesh_array[2]
        source_R = R_array[0]
        real_R = R_array[1]
        driving_R = R_array[2]
        source_t = t_array[0]
        real_t = t_array[1]
        driving_t = t_array[2]
        source_c = c_array[0]
        real_c = c_array[1]
        driving_c = c_array[2]
        source_mesh_image = mesh_img_array[0]
        real_mesh_image = mesh_img_array[1]
        driving_mesh_image = mesh_img_array[2]
        # source_mesh_image = mesh_img_array[0] * lip_mask_array[0]
        # real_mesh_image = mesh_img_array[1] * lip_mask_array[1]
        # driving_mesh_image = mesh_img_array[2] * lip_mask_array[2]
        source_z = z_array[0]
        real_z = z_array[1]
        driving_z = z_array[2]
        source_normed_z = normed_z_array[0]
        real_normed_z = normed_z_array[1]
        driving_normed_z = normed_z_array[2]

        video_array, prev_image = np.split(video_array, [len(video_array) - self.num_prevs], axis=0)
        mesh_img_array, prev_mesh_image = np.split(mesh_img_array, [len(mesh_img_array) - self.num_prevs], axis=0)

        prev_image = prev_image.transpose((0, 3, 1, 2))     # T x C x H x W
        prev_mesh_image = prev_mesh_image.transpose((0, 3, 1, 2))     # T x C x H x W

        out['driving'] = driving.transpose((2, 0, 1))
        out['real'] = real.transpose((2, 0, 1))
        out['source'] = source.transpose((2, 0, 1))
        out['driving_mesh'] = {'mesh': driving_mesh, 'normed_mesh': driving_normed_mesh, 'R': driving_R, 't': driving_t, 'c': driving_c, 'z': driving_z, 'normed_z': driving_normed_z}
        out['real_mesh'] = {'mesh': real_mesh, 'normed_mesh': real_normed_mesh, 'R': real_R, 't': real_t, 'c': real_c, 'z': real_z, 'normed_z': real_normed_z}
        out['source_mesh'] = {'mesh': source_mesh, 'normed_mesh': source_normed_mesh, 'R': source_R, 't': source_t, 'c': source_c, 'z': source_z, 'normed_z': source_normed_z}
        out['driving_mesh_image'] = driving_mesh_image.transpose((2, 0, 1))
        out['real_mesh_image'] = real_mesh_image.transpose((2, 0, 1))
        out['source_mesh_image'] = source_mesh_image.transpose((2, 0, 1))
        out['prev_image'] = prev_image.reshape((prev_image.shape[0] * prev_image.shape[1], prev_image.shape[2], prev_image.shape[3]))
        out['prev_mesh_image'] = prev_mesh_image[:, 0]
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
                 random_seed=0, pairs_list=None, augmentation_params=None):
        self.root_dir = root_dir
        self.videos = os.listdir(root_dir)
        self.frame_shape = tuple(frame_shape)
        self.pairs_list = pairs_list
        self.id_sampling = id_sampling
        self.reference_dict = torch.load('mesh_dict_reference.pt')
        if os.path.exists(os.path.join(root_dir, 'train')):
            assert os.path.exists(os.path.join(root_dir, 'test'))
            print("Use predefined train-test split.")
            if id_sampling:
                train_videos = {os.path.basename(video).split('#')[0] for video in
                                os.listdir(os.path.join(root_dir, 'train'))}
                train_videos = list(train_videos)
            else:
                train_videos = os.listdir(os.path.join(root_dir, 'train'))
            test_videos = os.listdir(os.path.join(root_dir, 'test'))
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
        return len(self.videos) // 5

    def __getitem__(self, idx):
        while True:
            try:
                idx = (idx + int(datetime.now().timestamp())) % len(self.videos) 
                if self.is_train and self.id_sampling:
                    name = self.videos[idx]
                    path = np.random.choice(glob.glob(os.path.join(self.root_dir, name)))
                else:
                    name = self.videos[idx]
                    path = os.path.join(self.root_dir, name)

                video_name = os.path.basename(path)

                if self.is_train and os.path.isdir(path):
                    frames = os.listdir(path)
                    num_frames = len(frames)
                    frame_idx = np.sort(np.random.choice(num_frames, replace=True, size=2))
                    frame_idx[1] = (frame_idx[1] + 100) % num_frames
                    raw_video_array = [io.imread(os.path.join(path, frames[(idx + int(datetime.now().timestamp())) % num_frames])) for idx in frame_idx]
                    video_array = np.stack([cv2.resize(img_as_float32(frame), self.frame_shape[:2]) for frame in raw_video_array], axis=0)
                    transform_hopenet =  transforms.Compose([
                                                            transforms.ToTensor(),
                                                            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
                    hopenet_video_array = [transform_hopenet(cv2.resize(frame, (224, 224))) for frame in raw_video_array]
                    
                else:
                    raw_video_array = read_video(path, frame_shape=self.frame_shape)
                    num_frames = len(raw_video_array)
                    frame_idx = np.sort(np.random.choice(num_frames, replace=True, size=2)) if self.is_train else range(
                        num_frames)
                    frames_idx =[((fid + int(datetime.now().timestamp())) % num_frames) for fid in frame_idx]
                    frames_idx[1] = (frames_idx[1] + 100) % num_frames
                    video_array = np.stack([cv2.resize(img_as_float32(raw_video_array[fid]), self.frame_shape[:2]) for fid in frame_idx], axis=0)
                
                
                    transform_hopenet =  transforms.Compose([
                                                            transforms.ToTensor(),
                                                            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
                    hopenet_video_array = [transform_hopenet(cv2.resize(raw_video_array[fid], (224, 224))) for fid in frame_idx]
                    
                if self.transform is not None:
                    video_array = self.transform(video_array)

                meshes = []
                for frame in video_array:
                    mesh = extract_mesh(img_as_ubyte(frame), self.reference_dict) # {value (N x 3), R (3 x 3), t(3 x 1), c1}
                    A = np.array([-1, -1, 1 / 2], dtype='float32')[:, np.newaxis] # 3 x 1
                    mesh['value'] = np.array(mesh['value'], dtype='float32') / 128 + np.squeeze(A, axis=-1)[None]
                    mesh['R'] = np.array(mesh['R'], dtype='float32')
                    mesh['c'] = np.array(mesh['c'], dtype='float32')
                    t = np.array(mesh['t'], dtype='float32')
                    mesh['t'] = (np.eye(3).astype(np.float32) - mesh['c'] * mesh['R']) @ A + t / 128
                    meshes.append(mesh)
                break
            except Exception as e:
                print(f'error: {e}')
                continue
                
        
        out = {}
        if self.is_train:
            source = np.array(video_array[0], dtype='float32')
            driving = np.array(video_array[1], dtype='float32')
            out['driving'] = driving.transpose((2, 0, 1))
            out['source'] = source.transpose((2, 0, 1))
            out['driving_mesh'] = meshes[1]
            out['source_mesh'] = meshes[0]
            out['hopenet_source'] = hopenet_video_array[0]
            out['hopenet_driving'] = hopenet_video_array[1]
        else:
            video = np.array(video_array, dtype='float32')
            out['video'] = video.transpose((3, 0, 1, 2))
            out['mesh'] = meshes
            
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
