import os, sys
import yaml
from argparse import ArgumentParser
from tqdm import tqdm

import warnings

warnings.filterwarnings(action='ignore')

import imageio
import numpy as np
from skimage.transform import resize
from skimage import img_as_ubyte, img_as_float32
import torch
import torch.nn.functional as F
from sync_batchnorm import DataParallelWithCallback

from modules.generator import OcclusionAwareGenerator, OcclusionAwareSPADEGenerator
from modules.headmodel import HeadModel
from modules.keypoint_detector import KPDetector, HEEstimator
from animate import normalize_kp
from scipy.spatial import ConvexHull
from modules.headmodel import HeadModel
<<<<<<< HEAD
from utils.util import extract_mesh, draw_section, draw_mouth_mask, get_mesh_image, matrix2euler, euler2matrix
=======
from utils.util import extract_mesh, draw_section, draw_mouth_mask, matrix2euler, euler2matrix, get_mesh_image
>>>>>>> a3f842d9c540e094d41883e5420257752a521610
from utils.one_euro_filter import OneEuroFilter
import cv2
import math

if sys.version_info[0] < 3:
    raise Exception("You must use Python 3 or higher. Recommended version is Python 3.7")

os.environ['CUDA_VISIBLE_DEVICES'] = '1'

def load_checkpoints(config, checkpoint_path, checkpoint_headmodel_path, checkpoint_posemodel_path, gen, cpu=False):

    headmodel = HeadModel(config['train_params'])
    
    if not cpu:
        headmodel.cuda()
    
    checkpoint_headmodel = torch.load(checkpoint_headmodel_path, map_location=torch.device('cpu' if cpu else 'cuda'))
    checkpoint_headmodel['headmodel'] = {k.replace('module.', ''): v for (k, v) in checkpoint_headmodel['headmodel'].items()}
    
    headmodel_dict= headmodel.state_dict()
    for k,v in checkpoint_headmodel['headmodel'].items():
        if k in headmodel_dict:
            headmodel_dict[k] = v
            
    headmodel.load_state_dict(headmodel_dict)
    headmodel.eval()
    
    statistics = headmodel.export_statistics()
    
    if gen == 'original':
        generator = OcclusionAwareGenerator(**config['model_params']['generator_params'],
                                            **config['model_params']['common_params'])
    elif gen == 'spade':
        generator = OcclusionAwareSPADEGenerator(**config['model_params']['generator_params'],
                                                 **config['model_params']['common_params'], headmodel=statistics)

    if not cpu:
        generator.cuda()

    if cpu:
        checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))
    else:
        checkpoint = torch.load(checkpoint_path)
 
    
    generator_dict= generator.state_dict()
    for k,v in checkpoint['generator'].items():
        if k in generator_dict:
            generator_dict[k] = v
            
    generator.load_state_dict(generator_dict)
    
    if not cpu:
        generator = DataParallelWithCallback(generator)

    generator.eval()


    posemodel = HEEstimator(**config['model_params']['he_estimator_params'],
                               **config['model_params']['common_params'])

    if torch.cuda.is_available():
        posemodel.cuda()

    ckpt_posemodel = torch.load(checkpoint_posemodel_path)
    posemodel.load_state_dict(ckpt_posemodel['he_estimator'])
    posemodel.eval()
    
    return generator, headmodel, posemodel



def headpose_pred_to_degree(pred):
    device = pred.device
    idx_tensor = [idx for idx in range(66)]
    idx_tensor = torch.FloatTensor(idx_tensor).to(device)
    pred = F.softmax(pred)
    degree = torch.sum(pred*idx_tensor, axis=1) * 3 - 99

    return degree



'''
# beta version
def get_rotation_matrix(yaw, pitch, roll):
    yaw = yaw / 180 * 3.14
    pitch = pitch / 180 * 3.14
    roll = roll / 180 * 3.14

    roll = roll.unsqueeze(1)
    pitch = pitch.unsqueeze(1)
    yaw = yaw.unsqueeze(1)

    roll_mat = torch.cat([torch.ones_like(roll), torch.zeros_like(roll), torch.zeros_like(roll), 
                          torch.zeros_like(roll), torch.cos(roll), -torch.sin(roll),
                          torch.zeros_like(roll), torch.sin(roll), torch.cos(roll)], dim=1)
    roll_mat = roll_mat.view(roll_mat.shape[0], 3, 3)

    pitch_mat = torch.cat([torch.cos(pitch), torch.zeros_like(pitch), torch.sin(pitch), 
                           torch.zeros_like(pitch), torch.ones_like(pitch), torch.zeros_like(pitch),
                           -torch.sin(pitch), torch.zeros_like(pitch), torch.cos(pitch)], dim=1)
    pitch_mat = pitch_mat.view(pitch_mat.shape[0], 3, 3)

    yaw_mat = torch.cat([torch.cos(yaw), -torch.sin(yaw), torch.zeros_like(yaw),  
                         torch.sin(yaw), torch.cos(yaw), torch.zeros_like(yaw),
                         torch.zeros_like(yaw), torch.zeros_like(yaw), torch.ones_like(yaw)], dim=1)
    yaw_mat = yaw_mat.view(yaw_mat.shape[0], 3, 3)

    rot_mat = torch.einsum('bij,bjk,bkm->bim', roll_mat, pitch_mat, yaw_mat)

    return rot_mat

'''

# def get_rotation_matrix(yaw, pitch, roll):
#     yaw = yaw / 180 * 3.14
#     pitch = pitch / 180 * 3.14
#     roll = roll / 180 * 3.14

#     roll = roll.unsqueeze(1)
#     pitch = pitch.unsqueeze(1)
#     yaw = yaw.unsqueeze(1)

#     pitch_mat = torch.cat([torch.ones_like(pitch), torch.zeros_like(pitch), torch.zeros_like(pitch), 
#                           torch.zeros_like(pitch), torch.cos(pitch), -torch.sin(pitch),
#                           torch.zeros_like(pitch), torch.sin(pitch), torch.cos(pitch)], dim=1)
#     pitch_mat = pitch_mat.view(pitch_mat.shape[0], 3, 3)

#     yaw_mat = torch.cat([torch.cos(yaw), torch.zeros_like(yaw), torch.sin(yaw), 
#                            torch.zeros_like(yaw), torch.ones_like(yaw), torch.zeros_like(yaw),
#                            -torch.sin(yaw), torch.zeros_like(yaw), torch.cos(yaw)], dim=1)
#     yaw_mat = yaw_mat.view(yaw_mat.shape[0], 3, 3)

#     roll_mat = torch.cat([torch.cos(roll), -torch.sin(roll), torch.zeros_like(roll),  
#                          torch.sin(roll), torch.cos(roll), torch.zeros_like(roll),
#                          torch.zeros_like(roll), torch.zeros_like(roll), torch.ones_like(roll)], dim=1)
#     roll_mat = roll_mat.view(roll_mat.shape[0], 3, 3)

#     rot_mat = torch.einsum('bij,bjk,bkm->bim', pitch_mat, yaw_mat, roll_mat)

#     return rot_mat


# def keypoint_transformation(kp_canonical, he, estimate_jacobian=True, free_view=False, yaw=0, pitch=0, roll=0):
#     kp = kp_canonical['value']
#     if not free_view:
#         yaw, pitch, roll = he['yaw'], he['pitch'], he['roll']
#         yaw = headpose_pred_to_degree(yaw)
#         pitch = headpose_pred_to_degree(pitch)
#         roll = headpose_pred_to_degree(roll)
#     else:
#         if yaw is not None:
#             yaw = torch.tensor([yaw]).cuda()
#         else:
#             yaw = he['yaw']
#             yaw = headpose_pred_to_degree(yaw)
#         if pitch is not None:
#             pitch = torch.tensor([pitch]).cuda()
#         else:
#             pitch = he['pitch']
#             pitch = headpose_pred_to_degree(pitch)
#         if roll is not None:
#             roll = torch.tensor([roll]).cuda()
#         else:
#             roll = he['roll']
#             roll = headpose_pred_to_degree(roll)

#     t, exp = he['t'], he['exp']

#     rot_mat = get_rotation_matrix(yaw, pitch, roll)
    
#     # keypoint rotation
#     kp_rotated = torch.einsum('bmp,bkp->bkm', rot_mat, kp)

#     # keypoint translation
#     t = t.unsqueeze_(1).repeat(1, kp.shape[1], 1)
#     kp_t = kp_rotated + t

#     # add expression deviation 
#     exp = exp.view(exp.shape[0], -1, 3)
#     kp_transformed = kp_t + exp

#     if estimate_jacobian:
#         jacobian = kp_canonical['jacobian']
#         jacobian_transformed = torch.einsum('bmp,bkps->bkms', rot_mat, jacobian)
#     else:
#         jacobian_transformed = None

#     return {'value': kp_transformed, 'jacobian': jacobian_transformed}

def preprocess_dict(d):
    res = {}
    for k, v in d.items():
        if type(v) == torch.Tensor:
            res[k] = v.cuda()[None]
        elif type(v) == list:
            res[k] = [v_i.cuda()[None] for v_i in v]
        elif type(v) == dict:
            res[k] = preprocess_dict(v)
            
    return res

def get_mesh_image_section(mesh, frame_shape, section_indices):
    # mesh: N0 x 3
    # print(f'sections shape: {sections.shape}')
    # mouth_mask = (255 * draw_mouth_mask(mesh[:, :2].numpy().astype(np.int32), frame_shape)).astype(np.int32)

    secs = draw_section(mesh[section_indices, :2].numpy().astype(np.int32), frame_shape, split=False) # (num_sections) x H x W x 3
    # print(f'draw section done')
    secs = torch.from_numpy(secs[:, :, :1].astype(np.float32).transpose((2, 0, 1)) / 255.0)
    # print('got mesh image sections')
    return secs
    
def calc_mesh_bias(mesh, head_statistics):
    sections = head_statistics['sections']
    mesh_bias = []
    for i, sec in enumerate(sections):
        mu_x = head_statistics[f'mu_x_{i}'] # N * 3
        bias = mesh['value'][:, sec[0]].mean(dim=1) # 1 x 3
        bias = bias.unsqueeze(1) + mu_x.view(1, -1, 3)
        mesh_bias.append(bias.flatten(1))
    return mesh_bias

def make_animation(source_image, driving_video, source_mesh, driving_meshes, generator, headmodel, relative=True, adapt_movement_scale=True, estimate_jacobian=True, cpu=False, free_view=False, yaw=0, pitch=0, roll=0):
    with torch.no_grad():
        predictions = []
        canonical = []
        exp = []
        
        headmodel_statistics = headmodel.export_statistics()
        
        source = torch.tensor(source_image[np.newaxis].astype(np.float32)).permute(0, 3, 1, 2)
        source_mesh = preprocess_dict(source_mesh)
        source_mesh['mesh_bias'] = calc_mesh_bias(source_mesh, headmodel_statistics)
        
        if not cpu:
            source = source.cuda()
            
        # driving = torch.tensor(np.array(driving_video)[np.newaxis].astype(np.float32)).permute(0, 4, 1, 2, 3)

        kp_source = source_mesh
        kp_driving_initial = preprocess_dict(driving_meshes[0])
        
        for frame_idx in tqdm(range(len(driving_meshes))):
            # driving_frame = driving[:, :, frame_idx]
            # if not cpu:
            #     driving_frame = driving_frame.cuda()
            kp_driving = preprocess_dict(driving_meshes[frame_idx])
            kp_driving['mesh_bias'] = calc_mesh_bias(kp_driving, headmodel_statistics)
            # print(f'before norm: {"mouth_img" in kp_driving}')
            kp_norm = kp_driving
            # kp_norm = normalize_kp(kp_source=kp_source, kp_driving=kp_driving,
            #                        kp_driving_initial=kp_driving_initial, use_relative_movement=relative,
            #                        use_relative_jacobian=estimate_jacobian, adapt_movement_scale=adapt_movement_scale)
            # print(f'after norm: {"mouth_img" in kp_norm}')

            out = generator(source, kp_source=kp_source, kp_driving=kp_norm)

            predictions.append(np.transpose(out['prediction'].data.cpu().numpy(), [0, 2, 3, 1])[0])
            # print(f'kp keys: {kp_source.keys()}')
            # print(f'kp drv keys: {kp_driving.keys()}')
            canonical.append(out['kp_source']['value'][0].data.cpu())
            exp.append(out['kp_driving']['value'][0].data.cpu())
            
    return {'prediction': predictions, 'kp': {'canonical': canonical, 'exp': exp}}


def find_best_frame(source, driving, cpu=False):
    import face_alignment

    def normalize_kp(kp):
        kp = kp - kp.mean(axis=0, keepdims=True)
        area = ConvexHull(kp[:, :2]).volume
        area = np.sqrt(area)
        kp[:, :2] = kp[:, :2] / area
        return kp

    fa = face_alignment.FaceAlignment(face_alignment.LandmarksType._2D, flip_input=True,
                                      device='cpu' if cpu else 'cuda')
    kp_source = fa.get_landmarks(255 * source)[0]
    kp_source = normalize_kp(kp_source)
    norm  = float('inf')
    frame_num = 0
    for i, image in tqdm(enumerate(driving)):
        kp_driving = fa.get_landmarks(255 * image)[0]
        kp_driving = normalize_kp(kp_driving)
        new_norm = (np.abs(kp_source - kp_driving) ** 2).sum()
        if new_norm < norm:
            norm = new_norm
            frame_num = i
    return frame_num


def adapt_values(origin, values, minimum=None, maximum=None, scale=None, center_align=False, center=None):
    # origin: float
    # values: tensor of size L
    sample_min, sample_max, sample_mean = values.min(), values.max(), values.mean()
    if (minimum is not None) and (maximum is not None):
        scale = min(scale, 1 / (sample_max - sample_min).clamp(min=1e-6))
    if not center_align:
        origin = sample_mean
    if center is not None:
        origin = center
    inter_values = origin + scale * torch.tanh(values - sample_mean)
    inter_min, inter_max = inter_values.min(), inter_values.max()
    adapted_values = inter_values
    
    if minimum is not None:
        clip = max(minimum, inter_min)
        delta = clip - inter_min
        adapted_values = adapted_values + delta
       
    if maximum is not None: 
        clip = min(maximum, inter_min)
        delta = clip - inter_min
        adapted_values = adapted_values + delta
    
    return adapted_values

def filter_values(values):
    MIN_CUTOFF = 1.0
    BETA = 1.0
    num_frames = len(values)
    fps = 25
    times = np.linspace(0, num_frames / fps, num_frames)
    
    filtered_values= []
    
    values = values * 100
    
    for i, x in enumerate(values):
        if i == 0:
            filter_value = OneEuroFilter(times[0], x, min_cutoff=MIN_CUTOFF, beta=BETA)
        else:
            x = filter_value(times[i], x)
        
        filtered_values.append(x)
        
    res = np.array(filtered_values)
    res = res / 100
    return res


def get_mouth_image(mesh, shape):
    mouth = draw_mouth_mask(mesh[:, :2].astype(np.int32), shape)
    mouth = torch.Tensor(mouth[:, :, :1].astype(np.float32).transpose((2, 0, 1)))
    
    return mouth
    
def filter_mesh(meshes, source_mesh):
    # meshes: list of dict of mesh({R, t, c})
    R_xs = []
    R_ys = []
    R_zs = []
    t_xs = []
    t_ys = []
    t_zs = []
    
    for i, mesh in enumerate(meshes):
        R, t, c = mesh['R'], mesh['t'], mesh['c']
        R_x, R_y, R_z = matrix2euler(R.numpy())
        # t_center = t - R @ t
        # t_x, t_y, t_z = t_center.squeeze(1)
        R_xs.append(R_x)
        R_ys.append(R_y)
        R_zs.append(R_z)
    
    R_xs = torch.tensor(R_xs).float()
    R_ys = torch.tensor(R_ys).float()
    R_zs = torch.tensor(R_zs).float()
    
    R_x_source, R_y_source, R_z_source = matrix2euler(source_mesh['R'].numpy())
    
    R_xs_adapted = adapt_values(R_x_source, R_xs, scale=(math.pi / 6), center_align=True)
    R_ys_adapted = adapt_values(R_y_source, R_ys, scale=(math.pi / 6), minimum=(-math.pi / 2), maximum=(math.pi / 2), center_align=True)
    R_zs_adapted = adapt_values(R_z_source, R_zs, scale=(math.pi / 6), minimum=(-math.pi / 4), maximum=(math.pi / 4), center_align=True)
    
    
    # R_xs_adapted = R_xs
    # R_ys_adapted = R_ys
    # R_zs_adapted = R_zs
    
    R_xs_adapted = torch.tensor(R_x_source)[None].repeat(len(R_xs))
    R_ys_adapted = torch.tensor(R_y_source)[None].repeat(len(R_ys))
    R_zs_adapted = torch.tensor(R_z_source)[None].repeat(len(R_zs))
    
    R_xs_filtered = torch.tensor(filter_values(R_xs_adapted.numpy())).float()
    R_ys_filtered = torch.tensor(filter_values(R_ys_adapted.numpy())).float()
    R_zs_filtered = torch.tensor(filter_values(R_zs_adapted.numpy())).float()
    
    # R_src, t_src = source_mesh['R'], source_mesh['t']
    # source_mesh['t'] = 
    Rs = []
    for R_x, R_y, R_z, mesh in zip(R_xs_filtered, R_ys_filtered, R_zs_filtered, meshes):
        R, t = mesh['R'], mesh['t']
        new_R = torch.tensor(euler2matrix([R_x, R_y, R_z])).float()
        Rs.append(new_R)
        mesh['R'] = new_R 
        t_center = new_R.inverse() @ R @ t
        # print(f't shape: {t.shape}')
        t_x, t_y, t_z = t_center.squeeze(1)
        t_xs.append(t_x)
        t_ys.append(t_y)
        t_zs.append(t_z)
    
    Rs = torch.stack(Rs, dim=0)
    
    t_xs = torch.tensor(t_xs).float()
    t_ys = torch.tensor(t_ys).float()
    t_zs = torch.tensor(t_zs).float()
    
    t_x_source, t_y_source, t_z_source = source_mesh['t'].squeeze(1)
    
    t_stack_raw = torch.stack([t_xs, t_ys, t_zs], dim=1)
    rot_raw = torch.einsum('bij,bj->bi', Rs.inverse(), t_stack_raw / source_mesh['c'])
    
    source_bias = torch.einsum('ij,jp->ip', source_mesh['R'].inverse(), - source_mesh['t'] / source_mesh['c']).squeeze(1)
    
    # rot_raw = rot_raw + 0.2 * (rot_raw - source_bias[None])
    
    t_xs_rot_raw = rot_raw[:, 0]
    t_ys_rot_raw = rot_raw[:, 1]
    t_zs_rot_raw = rot_raw[:, 2]
    
    t_xs_adapted = adapt_values(t_xs_rot_raw, t_xs_rot_raw, scale=0.5, center_align=False, center=0)
    t_ys_adapted = adapt_values(t_ys_rot_raw, t_ys_rot_raw, scale=0.5, center_align=False, center=0)
    t_zs_adapted = adapt_values(t_zs_rot_raw, t_zs_rot_raw, scale=0.5, center_align=False, center=-0.05)

    
    # t_xs_rot_filtered = torch.tensor(filter_values(t_xs_adapted.numpy())).float()
    # t_ys_rot_filtered = torch.tensor(filter_values(t_ys_adapted.numpy())).float()
    # t_zs_rot_filtered = torch.tensor(filter_values(t_zs_adapted.numpy())).float()
    
    # t_stack_rot_filtered = torch.stack([t_xs_rot_filtered, t_ys_rot_filtered, t_zs_rot_filtered], dim=1)
    # t_stack_filtered = torch.einsum('bij,bj->bi', Rs, t_stack_rot_filtered * source_mesh['c'])
    
    # t_xs_filtered = t_stack_filtered[:, 0]
    # t_ys_filtered = t_stack_filtered[:, 1]
    # t_zs_filtered = t_stack_filtered[:, 2]
    

    t_xs_adapted = t_x_source[None].repeat(len(t_xs))
    t_ys_adapted = t_y_source[None].repeat(len(t_ys))
    t_zs_adapted = t_z_source[None].repeat(len(t_zs))
    
    t_xs_filtered = t_xs_adapted
    t_ys_filtered = t_ys_adapted
    t_zs_filtered = t_zs_adapted
    
    for t_x, t_y, t_z, mesh in zip(t_xs_filtered, t_ys_filtered, t_zs_filtered, meshes):
        # t_x, t_y, t_z = mesh['t'].squeeze(1)
        new_t = torch.tensor([torch.tensor(t_x), torch.tensor(t_y), torch.tensor(t_z)]).unsqueeze(1)
        # new_t = source_mesh['t']
        # print(f'delta t: {new_t - mesh["t"]}')
        mesh['t'] = new_t
        mesh['c'] = source_mesh['c']
    
    # t_stack = torch.stack([t_xs_filtered, t_ys_filtered, t_zs_filtered], dim=1)
    # rot = torch.einsum('bij,bj->bi', Rs.inverse(), t_stack / source_mesh['c'])
    
    # torch.save(t_stack_rot_filtered, os.path.join(opt.result_dir, f'{opt.result_video}_zs.pt'))

        
if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--config", default='config/vox-256.yaml', help="path to config")
    parser.add_argument("--checkpoint", default='', help="path to checkpoint to restore")
    parser.add_argument("--checkpoint_headmodel", default='', help="path to headmodel checkpoint to restore")
    parser.add_argument("--checkpoint_posemodel", default='/home/ubuntu/workspace/BFv2v/ckpt/00000189-checkpoint.pth.tar', help="path to he_estimator checkpoint")
    
    parser.add_argument("--reference_dict", default='mesh_dict_reference.pt', help="path to reference dict to restore")

    parser.add_argument("--source_image", default='', help="path to source image")
    parser.add_argument("--driving_video", default='', help="path to driving video")
    parser.add_argument("--result_video", default='result.mp4', help="path to output")
    parser.add_argument("--result_dir", default='result', help="path to result dir")
    parser.add_argument("--driven_dir", default=None, help="path to driven data dir")

    parser.add_argument("--gen", default="spade", choices=["original", "spade"])
    parser.add_argument("--ignore_emotion", action='store_true')
 
    parser.add_argument("--relative", dest="relative", action="store_true", help="use relative or absolute keypoint coordinates")
    parser.add_argument("--adapt_scale", dest="adapt_scale", action="store_true", help="adapt movement scale based on convex hull of keypoints")

    parser.add_argument("--find_best_frame", dest="find_best_frame", action="store_true", 
                        help="Generate from the frame that is the most alligned with source. (Only for faces, requires face_aligment lib)")

    parser.add_argument("--best_frame", dest="best_frame", type=int, default=None,  
                        help="Set frame to start from.")
 
    parser.add_argument("--cpu", dest="cpu", action="store_true", help="cpu mode.")

    parser.add_argument("--free_view", dest="free_view", action="store_true", help="control head pose")
    parser.add_argument("--yaw", dest="yaw", type=int, default=None, help="yaw")
    parser.add_argument("--pitch", dest="pitch", type=int, default=None, help="pitch")
    parser.add_argument("--roll", dest="roll", type=int, default=None, help="roll")
 

    parser.set_defaults(relative=False)
    parser.set_defaults(adapt_scale=False)
    parser.set_defaults(free_view=False)

    opt = parser.parse_args()

    with open(opt.config) as f:
        config = yaml.load(f)

    config['train_params']['num_kp'] = config['model_params']['common_params']['num_kp']
    config['train_params']['sections'] = config['model_params']['common_params']['sections']
    config['train_params']['headmodel_sections'] = config['model_params']['common_params']['headmodel_sections']
    sections = config['train_params']['sections']
    
    generator, headmodel, posemodel = load_checkpoints(config=config, checkpoint_path=opt.checkpoint, checkpoint_headmodel_path=opt.checkpoint_headmodel, checkpoint_posemodel_path=opt.checkpoint_posemodel, gen=opt.gen, cpu=opt.cpu)
    generator.ignore_emotion = opt.ignore_emotion
    
    print(f'Generator Ignorance: {generator.ignore_emotion}')
    estimate_jacobian = config['model_params']['common_params']['estimate_jacobian']
    print(f'estimate jacobian: {estimate_jacobian}')

    reference_dict = torch.load(opt.reference_dict)
    source_image = imageio.imread(opt.source_image)
    fps = 25
    driving_video = []
    section_indices = []
    for sec in sections:
        section_indices.extend(sec[0])
    print(f'len section indices: {len(section_indices)}')
    section_mouth = sections[-1][0]
    
    if len(opt.driving_video) > 0:
        reader = imageio.get_reader(opt.driving_video)
        try:
            for im in reader:
                driving_video.append(im)
        except RuntimeError:
            pass
        reader.close()

    frame_shape = config['dataset_params']['frame_shape']

    if len(source_image.shape) == 2:
        source_image = cv2.cvtColor(source_image, cv2.COLOR_GRAY2RGB)
    source_image = resize(img_as_float32(source_image), frame_shape[:2])[..., :3]
    driving_video = [resize(img_as_float32(frame), frame_shape[:2])[..., :3] for frame in driving_video]
    
    L = frame_shape[0]
    mesh = extract_mesh(img_as_ubyte(source_image), reference_dict)
    A = np.array([-1, -1, 1 / 2], dtype='float32')[:, np.newaxis] # 3 x 1
    mesh['value'] = torch.from_numpy(np.array(mesh['value'], dtype='float32') * 2 / L  + np.squeeze(A, axis=-1)[None])
    mesh['R'] = torch.from_numpy(np.array(mesh['R'], dtype='float32'))
    mesh['c'] = torch.from_numpy(np.array(mesh['c'], dtype='float32'))
    t = np.array(mesh['t'], dtype='float32')
    mesh['t'] = torch.from_numpy((np.eye(3).astype(np.float32) - mesh['c'].numpy() * mesh['R'].numpy()) @ A + t * 2 / L)
    source_mesh = mesh
    
    static_source_mesh = (1 / mesh['c'][None, None]) * torch.einsum('ij,nj->ni', mesh['R'].inverse(), source_mesh['value'][section_indices].cpu() - mesh['t'][None, :, 0])
    static_source_mesh = L * (static_source_mesh - torch.from_numpy(np.squeeze(A, axis=-1)[None])) // 2  
    
    driving_meshes = []
    inter_meshes = []
    target_meshes = []

    if opt.driven_dir is None:
        print("Using Given Driving Video...")
        for frame in driving_video:
            mesh = extract_mesh(img_as_ubyte(frame), reference_dict)
            A = np.array([-1, -1, 1 / 2], dtype='float32')[:, np.newaxis] # 3 x 1
            mesh['value'] = torch.from_numpy(np.array(mesh['value'], dtype='float32') * 2 / L  + np.squeeze(A, axis=-1)[None])
            mesh['R'] = torch.from_numpy(np.array(mesh['R'], dtype='float32'))
            mesh['c'] = torch.from_numpy(np.array(mesh['c'], dtype='float32'))
            t = np.array(mesh['t'], dtype='float32')
            mesh['t'] = torch.from_numpy((np.eye(3).astype(np.float32) - mesh['c'].numpy() * mesh['R'].numpy()) @ A + t * 2 / L)
            
            mesh['R'] = torch.tensor(source_mesh['R'])
            mesh['c'] = torch.tensor(source_mesh['c'])
            mesh['t'] = torch.tensor(source_mesh['t'])
            target_mesh = (1 / source_mesh['c'][None, None]) * torch.einsum('ij,nj->ni', source_mesh['R'].inverse(), mesh['value'][section_indices].cpu() - source_mesh['t'][None, :, 0])
            target_mesh = L * (target_mesh - torch.from_numpy(np.squeeze(A, axis=-1)[None])) // 2
            mesh['intermediate_mesh_img_sec'] = get_mesh_image_section(target_mesh, frame_shape)
            driving_meshes.append(mesh)    

    else:
        print("Using Pre-driven Data...")
        source_mesh = torch.load(os.path.join(opt.driven_dir, 'result', 'source_mesh.pt'))
        raw_mesh = L * (source_mesh['raw_value'] - torch.from_numpy(np.squeeze(A, axis=-1)[None])) // 2
        source_mesh['mesh_img_sec'] = get_mesh_image_section(raw_mesh, frame_shape, section_indices)
        driven_meshes = torch.load(os.path.join(opt.driven_dir, 'result', 'driven_meshes.pt'))
        for driven_mesh in driven_meshes:
            mesh = {}

            mesh['_value'] = driven_mesh['value']
            mesh['value'] = torch.tensor(source_mesh['value'])
            mesh['raw_value'] = driven_mesh['raw_value']
            mesh['fake_raw_value'] = driven_mesh['fake_raw_value']
            mesh['R'] = driven_mesh['R']
            mesh['c'] = driven_mesh['c']
            mesh['t'] = driven_mesh['t']
            mesh['he_R'] = driven_mesh['he_R']
            mesh['he_t'] = driven_mesh['he_t']
            mesh['he_value'] = driven_mesh['he_value']
            mesh['he_raw_value'] = driven_mesh['he_raw_value']
            mesh['he_bias'] = driven_mesh['he_bias']
            # mesh['R'] = source_mesh['R'].cpu()
            # mesh['c'] = source_mesh['c'].cpu()
            # mesh['t'] = source_mesh['t'].cpu()
            
            # driven_mesh['driven_sections'][:-len(section_mouth)] = torch.tensor(mesh['value'][section_indices][:-len(section_mouth)])
            mesh['value'][section_indices] = driven_mesh['driven_sections'][:len(section_indices)]
            # print(f'delta mesh check: {(driven_mesh["driven_sections"].cpu() - source_mesh["value"][section_indices].cpu()).norm()}')
            target_mesh = (1 / source_mesh['c'][None, None]) * torch.einsum('ij,nj->ni', source_mesh['R'].inverse(), driven_mesh['driven_sections'].cpu() - source_mesh['t'][None, :, 0])
            target_mesh = L * (target_mesh - torch.from_numpy(np.squeeze(A, axis=-1)[None])) // 2
            # mesh['intermediate_mesh_img_sec'] = get_mesh_image_section(target_mesh, frame_shape)
            driving_meshes.append(mesh)
            # inter_meshes.append(target_mesh)
            # target_mesh = (1 / mesh['c'][None, None]) * torch.einsum('ij,nj->ni', mesh['R'].inverse(), driven_mesh['driven_sections'].cpu() - mesh['t'][None, :, 0])
            # target_mesh = L * (target_mesh - torch.from_numpy(np.squeeze(A, axis=-1)[None])) // 2
            # target_meshes.append(target_mesh)




    # use one euro filter for denoising
    # filter_mesh(driving_meshes, source_mesh)
    target_meshes = []
    for mesh in driving_meshes:
        raw_mesh = mesh['he_raw_value']
        # raw_mesh = (1 / mesh['c'][None, None]) * torch.einsum('ij,nj->ni', mesh['R'].inverse(), mesh['value'].cpu() - mesh['t'][None, :, 0])
        # print('msh img sec got')
        raw_mesh = L * (raw_mesh - torch.from_numpy(np.squeeze(A, axis=-1)[None])) // 2

        fake_raw_mesh = mesh['fake_raw_value']
        fake_raw_mesh = L * (fake_raw_mesh - torch.from_numpy(np.squeeze(A, axis=-1)[None])) // 2
<<<<<<< HEAD
=======
        print(f"mesh difference: {(source_mesh['value'] - mesh['value'])[section_indices]}")
>>>>>>> a3f842d9c540e094d41883e5420257752a521610
        mesh['fake_mesh_img'] = torch.from_numpy((get_mesh_image(fake_raw_mesh, frame_shape)[:, :, [0]] / 255.0).transpose((2, 0, 1))).float()
        # mesh['mouth_img'] = get_mouth_image(raw_mesh.numpy(),  frame_shape)
        mesh['mesh_img_sec'] = get_mesh_image_section(raw_mesh, frame_shape, section_indices)
        # print(f'mouth image shape: {mesh["mouth_img"].shape}')
        target_meshes.append(fake_raw_mesh[section_indices])
        # mesh['raw_value'] = np.array(raw_mesh, dtype='float32') * 2 / L + np.squeeze(A, axis=-1)[None]
        

    if opt.find_best_frame or opt.best_frame is not None:
        i = opt.best_frame if opt.best_frame is not None else find_best_frame(source_image, driving_video, cpu=opt.cpu)
        print ("Best frame: " + str(i))
        driving_forward = driving_video[i:]
        driving_backward = driving_video[:(i+1)][::-1]
        driving_meshes_forward = driving_meshes[i:]
        driving_meshes_backward = driving_meshes[:(i+1)][::-1]
        output_forward = make_animation(source_image, driving_forward, source_mesh, driving_meshes_forward, generator, headmodel, relative=opt.relative, adapt_movement_scale=opt.adapt_scale, estimate_jacobian=estimate_jacobian, cpu=opt.cpu, free_view=opt.free_view, yaw=opt.yaw, pitch=opt.pitch, roll=opt.roll, ignore_emotion=ignore_emotion)
        output_backward = make_animation(source_image, driving_backward, source_mesh, driving_meshes_backward, generator, headmodel, relative=opt.relative, adapt_movement_scale=opt.adapt_scale, estimate_jacobian=estimate_jacobian, cpu=opt.cpu, free_view=opt.free_view, yaw=opt.yaw, pitch=opt.pitch, roll=opt.roll, ignore_emotion=eignore_emotion)
        predictions = output_backward['prediction'][::-1] + output_forward['prediction'][1:]
        kps = {k: (output_backward['kp'][k][::-1] + output_forward['kp'][k][1:]) for k in ('canonical', 'exp')}
    else:
        output = make_animation(source_image, driving_video, source_mesh, driving_meshes, generator, headmodel, relative=opt.relative, adapt_movement_scale=opt.adapt_scale, estimate_jacobian=estimate_jacobian, cpu=opt.cpu, free_view=opt.free_view, yaw=opt.yaw, pitch=opt.pitch, roll=opt.roll)
        predictions = output['prediction']
        kps = output['kp'] 
    
    print(f'result video name: {opt.result_video}')
    
    # mesh styling
    meshed_frames = []
    for i, frame in enumerate(predictions):
        mesh = target_meshes[i]
        # print(f'frame type: {img_as_ubyte(frame).dtype}')
        frame = np.ascontiguousarray(img_as_ubyte(frame))
        # meshed_frame = draw_section(mesh[:, :2].numpy().astype(np.int32), frame_shape, mask=frame)
        meshed_frame = frame
        # meshed_frame = (255 * driving_meshes[i]['mouth_img'].repeat(3, 1, 1).permute(1, 2, 0)).numpy().astype(np.int32)
        meshed_frames.append(meshed_frame)

    imageio.mimsave(os.path.join(opt.result_dir, opt.result_video), meshed_frames, fps=fps)

    # imageio.mimsave(os.path.join(opt.result_dir, 'video', opt.result_video), [img_as_ubyte(frame) for frame in predictions], fps=fps)
    # torch.save(kps, os.path.join(opt.result_dir, 'data', opt.result_video + '.pt'))
    
    reader = imageio.get_reader(opt.driving_video)
    try:
        for im in reader:
            driving_video.append(im)
    except RuntimeError:
        pass
    reader.close()

    frame_shape = config['dataset_params']['frame_shape']
    
    driving_video = [resize(frame, frame_shape[:2])[..., :3] for frame in driving_video]
    meshed_frames = []
    for i, frame in enumerate(driving_video):
        if i >= len(target_meshes):
            continue
        mesh = driving_meshes[i]['value']
        mesh = L * (mesh - torch.from_numpy(np.squeeze(A, axis=-1)[None])) // 2
        # print(f'_mesh shape: {mesh.shape}')
        # mesh = target_meshes[i]
        meshed_frame = draw_section(mesh[:, :2].numpy().astype(np.int32), frame_shape, mask=img_as_ubyte(frame))
        # meshed_frame = img_as_ubyte(frame)
        meshed_frames.append(meshed_frame)
        
    
    imageio.mimsave(os.path.join(opt.result_dir, '00001_raw_mesh.mp4'), meshed_frames, fps=fps)

    meshed_frames = []
    for i, frame in enumerate(driving_video):
        if i >= len(target_meshes):
            continue
        mesh = driving_meshes[i]['value']
        mesh = L * (mesh - torch.from_numpy(np.squeeze(A, axis=-1)[None])) // 2
        # print(f'_mesh shape: {mesh.shape}')
        # mesh = target_meshes[i]
        meshed_frame = draw_section(mesh[:, :2].numpy().astype(np.int32), frame_shape, mask=img_as_ubyte(frame))
        # meshed_frame = img_as_ubyte(frame)
        meshed_frames.append(meshed_frame)
        
    
    imageio.mimsave(os.path.join(opt.result_dir, '00001_normed_mesh.mp4'), meshed_frames, fps=fps)


    # meshed_frames = []
    # for i in range(len(driving_video)):
    #     # if i >= len(inter_meshes):
    #     #     continue
    #     mesh = static_source_mesh
    #     # meshed_frame = draw_section(mesh[:, :2].numpy().astype(np.int32), frame_shape, mask=img_as_ubyte(source_image))
    #     meshed_frame = img_as_ubyte(source_image)
    #     meshed_frames.append(meshed_frame)
        
    # imageio.mimsave(os.path.join(opt.result_dir, 'E_00002_meshed.mp4'), meshed_frames, fps=fps)

    