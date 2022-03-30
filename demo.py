import os, sys
import yaml
from argparse import ArgumentParser
from tqdm import tqdm

import warnings

warnings.filterwarnings(action='ignore')

import imageio
import numpy as np
from skimage.transform import resize
from skimage import img_as_ubyte
import torch
import torch.nn.functional as F
from sync_batchnorm import DataParallelWithCallback

from modules.generator import OcclusionAwareGenerator, OcclusionAwareSPADEGenerator
from modules.headmodel import HeadModel
from modules.keypoint_detector import KPDetector, HEEstimator
from animate import normalize_kp
from scipy.spatial import ConvexHull
from modules.headmodel import HeadModel
from utils.util import extract_mesh

if sys.version_info[0] < 3:
    raise Exception("You must use Python 3 or higher. Recommended version is Python 3.7")

os.environ['CUDA_VISIBLE_DEVICES'] = '3'

def load_checkpoints(config, checkpoint_path, checkpoint_headmodel_path, gen, cpu=False):

    if gen == 'original':
        generator = OcclusionAwareGenerator(**config['model_params']['generator_params'],
                                            **config['model_params']['common_params'])
    elif gen == 'spade':
        generator = OcclusionAwareSPADEGenerator(**config['model_params']['generator_params'],
                                                 **config['model_params']['common_params'])

    if not cpu:
        generator.cuda()

    if cpu:
        checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))
    else:
        checkpoint = torch.load(checkpoint_path)
 
    generator.load_state_dict(checkpoint['generator'])
    
    if not cpu:
        generator = DataParallelWithCallback(generator)

    generator.eval()
    
    headmodel = HeadModel(config['train_params'])
    
    if not cpu:
        headmodel.cuda()
    
    checkpoint_headmodel = torch.load(checkpoint_headmodel_path, map_location=torch.device('cpu' if cpu else 'cuda'))
    checkpoint_headmodel['headmodel'] = {k.replace('module.', ''): v for (k, v) in checkpoint_headmodel['headmodel'].items()}
    headmodel.load_state_dict(checkpoint_headmodel['headmodel'])
    
    headmodel.eval()
        
    return generator, headmodel



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
    return {k: v.cuda()[None] for (k, v) in d.items()}

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
            
        driving = torch.tensor(np.array(driving_video)[np.newaxis].astype(np.float32)).permute(0, 4, 1, 2, 3)

        kp_source = source_mesh
        kp_driving_initial = preprocess_dict(driving_meshes[0])
        
        for frame_idx in tqdm(range(driving.shape[2])):
            driving_frame = driving[:, :, frame_idx]
            if not cpu:
                driving_frame = driving_frame.cuda()
            kp_driving = preprocess_dict(driving_meshes[frame_idx])
            kp_driving['mesh_bias'] = calc_mesh_bias(kp_driving, headmodel_statistics)
            kp_norm = normalize_kp(kp_source=kp_source, kp_driving=kp_driving,
                                   kp_driving_initial=kp_driving_initial, use_relative_movement=relative,
                                   use_relative_jacobian=estimate_jacobian, adapt_movement_scale=adapt_movement_scale)
            out = generator(source, kp_source=kp_source, kp_driving=kp_norm)

            predictions.append(np.transpose(out['prediction'].data.cpu().numpy(), [0, 2, 3, 1])[0])
            # print(f'kp keys: {kp_source.keys()}')
            # print(f'kp drv keys: {kp_driving.keys()}')
            canonical.append(out['kp_source'][0].data.cpu())
            exp.append(out['kp_driving'][0].data.cpu())
            
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

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--config", default='config/vox-256.yaml', help="path to config")
    parser.add_argument("--checkpoint", default='', help="path to checkpoint to restore")
    parser.add_argument("--checkpoint_headmodel", default='', help="path to headmodel checkpoint to restore")
    parser.add_argument("--reference_dict", default='', help="path to reference dict to restore")

    parser.add_argument("--source_image", default='', help="path to source image")
    parser.add_argument("--driving_video", default='', help="path to driving video")
    parser.add_argument("--result_video", default='result.mp4', help="path to output")
    parser.add_argument("--result_dir", default='result', help="path to result dir")

    parser.add_argument("--gen", default="spade", choices=["original", "spade"])
 
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
    
    reference_dict = torch.load(opt.reference_dict)
    source_image = imageio.imread(opt.source_image)
    reader = imageio.get_reader(opt.driving_video)
    fps = reader.get_meta_data()['fps']
    driving_video = []
    try:
        for im in reader:
            driving_video.append(im)
    except RuntimeError:
        pass
    reader.close()

    frame_shape = config['dataset_params']['frame_shape']
    
    source_image = resize(source_image, frame_shape[:2])[..., :3]
    driving_video = [resize(frame, frame_shape[:2])[..., :3] for frame in driving_video]
    
    L = frame_shape[0]
    mesh = extract_mesh(img_as_ubyte(source_image), reference_dict)
    A = np.array([-1, -1, 1 / 2], dtype='float32')[:, np.newaxis] # 3 x 1
    mesh['value'] = torch.from_numpy(np.array(mesh['value'], dtype='float32') * 2 / L  + np.squeeze(A, axis=-1)[None])
    mesh['R'] = torch.from_numpy(np.array(mesh['R'], dtype='float32'))
    mesh['c'] = torch.from_numpy(np.array(mesh['c'], dtype='float32'))
    t = np.array(mesh['t'], dtype='float32')
    mesh['t'] = torch.from_numpy((np.eye(3).astype(np.float32) - mesh['c'].numpy() * mesh['R'].numpy()) @ A + t * 2 / L)
    source_mesh = mesh
    
    driving_meshes = []
    for frame in driving_video:
        mesh = extract_mesh(img_as_ubyte(frame), reference_dict)
        A = np.array([-1, -1, 1 / 2], dtype='float32')[:, np.newaxis] # 3 x 1
        mesh['value'] = torch.from_numpy(np.array(mesh['value'], dtype='float32') * 2 / L  + np.squeeze(A, axis=-1)[None])
        mesh['R'] = torch.from_numpy(np.array(mesh['R'], dtype='float32'))
        mesh['c'] = torch.from_numpy(np.array(mesh['c'], dtype='float32'))
        t = np.array(mesh['t'], dtype='float32')
        mesh['t'] = torch.from_numpy((np.eye(3).astype(np.float32) - mesh['c'].numpy() * mesh['R'].numpy()) @ A + t * 2 / L)
        driving_meshes.append(mesh)    
        
    generator, headmodel = load_checkpoints(config=config, checkpoint_path=opt.checkpoint, checkpoint_headmodel_path=opt.checkpoint_headmodel, gen=opt.gen, cpu=opt.cpu)


        
    estimate_jacobian = config['model_params']['common_params']['estimate_jacobian']
    print(f'estimate jacobian: {estimate_jacobian}')

    if opt.find_best_frame or opt.best_frame is not None:
        i = opt.best_frame if opt.best_frame is not None else find_best_frame(source_image, driving_video, cpu=opt.cpu)
        print ("Best frame: " + str(i))
        driving_forward = driving_video[i:]
        driving_backward = driving_video[:(i+1)][::-1]
        driving_meshes_forward = driving_meshes[i:]
        driving_meshes_backward = driving_meshes[:(i+1)][::-1]
        output_forward = make_animation(source_image, driving_forward, source_mesh, driving_meshes_forward, generator, headmodel, relative=opt.relative, adapt_movement_scale=opt.adapt_scale, estimate_jacobian=estimate_jacobian, cpu=opt.cpu, free_view=opt.free_view, yaw=opt.yaw, pitch=opt.pitch, roll=opt.roll)
        output_backward = make_animation(source_image, driving_backward, source_mesh, driving_meshes_backward, generator, headmodel, relative=opt.relative, adapt_movement_scale=opt.adapt_scale, estimate_jacobian=estimate_jacobian, cpu=opt.cpu, free_view=opt.free_view, yaw=opt.yaw, pitch=opt.pitch, roll=opt.roll)
        predictions = output_backward['prediction'][::-1] + output_forward['prediction'][1:]
        kps = {k: (output_backward['kp'][k][::-1] + output_forward['kp'][k][1:]) for k in ('canonical', 'exp')}
    else:
        output = make_animation(source_image, driving_video, source_mesh, driving_meshes, generator, headmodel, relative=opt.relative, adapt_movement_scale=opt.adapt_scale, estimate_jacobian=estimate_jacobian, cpu=opt.cpu, free_view=opt.free_view, yaw=opt.yaw, pitch=opt.pitch, roll=opt.roll)
        predictions = output['prediction']
        kps = output['kp'] 
    
    print(f'result video name: {opt.result_video}')
    imageio.mimsave(os.path.join(opt.result_dir, 'video', opt.result_video), [img_as_ubyte(frame) for frame in predictions], fps=fps)
    torch.save(kps, os.path.join(opt.result_dir, 'data', opt.result_video + '.pt'))