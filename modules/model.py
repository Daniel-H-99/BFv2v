from torch import nn
import torch
import torch.nn.functional as F
from modules.util import AntiAliasInterpolation2d, make_coordinate_grid_2d
from torchvision import models
import numpy as np
from torch.autograd import grad
import modules.hopenet as hopenet
from torchvision import transforms
import math

class Vgg19(torch.nn.Module):
    """
    Vgg19 network for perceptual loss.
    """
    def __init__(self, requires_grad=False):
        super(Vgg19, self).__init__()
        vgg_pretrained_features = models.vgg19(pretrained=True).features
        self.slice1 = torch.nn.Sequential()
        self.slice2 = torch.nn.Sequential()
        self.slice3 = torch.nn.Sequential()
        self.slice4 = torch.nn.Sequential()
        self.slice5 = torch.nn.Sequential()
        for x in range(2):
            self.slice1.add_module(str(x), vgg_pretrained_features[x])
        for x in range(2, 7):
            self.slice2.add_module(str(x), vgg_pretrained_features[x])
        for x in range(7, 12):
            self.slice3.add_module(str(x), vgg_pretrained_features[x])
        for x in range(12, 21):
            self.slice4.add_module(str(x), vgg_pretrained_features[x])
        for x in range(21, 30):
            self.slice5.add_module(str(x), vgg_pretrained_features[x])

        self.mean = torch.nn.Parameter(data=torch.Tensor(np.array([0.485, 0.456, 0.406]).reshape((1, 3, 1, 1))),
                                       requires_grad=False)
        self.std = torch.nn.Parameter(data=torch.Tensor(np.array([0.229, 0.224, 0.225]).reshape((1, 3, 1, 1))),
                                      requires_grad=False)

        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, X):
        X = (X - self.mean) / self.std
        h_relu1 = self.slice1(X)
        h_relu2 = self.slice2(h_relu1)
        h_relu3 = self.slice3(h_relu2)
        h_relu4 = self.slice4(h_relu3)
        h_relu5 = self.slice5(h_relu4)
        out = [h_relu1, h_relu2, h_relu3, h_relu4, h_relu5]
        return out


class ImagePyramide(torch.nn.Module):
    """
    Create image pyramide for computing pyramide perceptual loss.
    """
    def __init__(self, scales, num_channels):
        super(ImagePyramide, self).__init__()
        downs = {}
        for scale in scales:
            downs[str(scale).replace('.', '-')] = AntiAliasInterpolation2d(num_channels, scale)
        self.downs = nn.ModuleDict(downs)

    def forward(self, x):
        out_dict = {}
        for scale, down_module in self.downs.items():
            out_dict['prediction_' + str(scale).replace('-', '.')] = down_module(x)
        return out_dict


class Transform:
    """
    Random tps transformation for equivariance constraints.
    """
    def __init__(self, bs, **kwargs):
        noise = torch.normal(mean=0, std=kwargs['sigma_affine'] * torch.ones([bs, 2, 3]))
        self.theta = noise + torch.eye(2, 3).view(1, 2, 3)
        self.bs = bs

        if ('sigma_tps' in kwargs) and ('points_tps' in kwargs):
            self.tps = True
            self.control_points = make_coordinate_grid_2d((kwargs['points_tps'], kwargs['points_tps']), type=noise.type())
            self.control_points = self.control_points.unsqueeze(0)
            self.control_params = torch.normal(mean=0,
                                               std=kwargs['sigma_tps'] * torch.ones([bs, 1, kwargs['points_tps'] ** 2]))
        else:
            self.tps = False

    def transform_frame(self, frame):
        grid = make_coordinate_grid_2d(frame.shape[2:], type=frame.type()).unsqueeze(0)
        grid = grid.view(1, frame.shape[2] * frame.shape[3], 2)
        grid = self.warp_coordinates(grid).view(self.bs, frame.shape[2], frame.shape[3], 2)
        return F.grid_sample(frame, grid, padding_mode="reflection")

    def warp_coordinates(self, coordinates):
        theta = self.theta.type(coordinates.type())
        theta = theta.unsqueeze(1)
        transformed = torch.matmul(theta[:, :, :, :2], coordinates.unsqueeze(-1)) + theta[:, :, :, 2:]
        transformed = transformed.squeeze(-1)

        if self.tps:
            control_points = self.control_points.type(coordinates.type())
            control_params = self.control_params.type(coordinates.type())
            distances = coordinates.view(coordinates.shape[0], -1, 1, 2) - control_points.view(1, 1, -1, 2)
            distances = torch.abs(distances).sum(-1)

            result = distances ** 2
            result = result * torch.log(distances + 1e-6)
            result = result * control_params
            result = result.sum(dim=2).view(self.bs, coordinates.shape[1], 1)
            transformed = transformed + result

        return transformed

    def jacobian(self, coordinates):
        new_coordinates = self.warp_coordinates(coordinates)
        grad_x = grad(new_coordinates[..., 0].sum(), coordinates, create_graph=True)
        grad_y = grad(new_coordinates[..., 1].sum(), coordinates, create_graph=True)
        jacobian = torch.cat([grad_x[0].unsqueeze(-2), grad_y[0].unsqueeze(-2)], dim=-2)
        return jacobian


def detach_kp(kp):
    return {key: value.detach() for key, value in kp.items()}


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

def get_rotation_matrix(yaw, pitch, roll):
    yaw = yaw / 180 * 3.14
    pitch = pitch / 180 * 3.14
    roll = roll / 180 * 3.14

    roll = roll.unsqueeze(1)
    pitch = pitch.unsqueeze(1)
    yaw = yaw.unsqueeze(1)

    pitch_mat = torch.cat([torch.ones_like(pitch), torch.zeros_like(pitch), torch.zeros_like(pitch), 
                          torch.zeros_like(pitch), torch.cos(pitch), -torch.sin(pitch),
                          torch.zeros_like(pitch), torch.sin(pitch), torch.cos(pitch)], dim=1)
    pitch_mat = pitch_mat.view(pitch_mat.shape[0], 3, 3)

    yaw_mat = torch.cat([torch.cos(yaw), torch.zeros_like(yaw), torch.sin(yaw), 
                           torch.zeros_like(yaw), torch.ones_like(yaw), torch.zeros_like(yaw),
                           -torch.sin(yaw), torch.zeros_like(yaw), torch.cos(yaw)], dim=1)
    yaw_mat = yaw_mat.view(yaw_mat.shape[0], 3, 3)

    roll_mat = torch.cat([torch.cos(roll), -torch.sin(roll), torch.zeros_like(roll),  
                         torch.sin(roll), torch.cos(roll), torch.zeros_like(roll),
                         torch.zeros_like(roll), torch.zeros_like(roll), torch.ones_like(roll)], dim=1)
    roll_mat = roll_mat.view(roll_mat.shape[0], 3, 3)

    rot_mat = torch.einsum('bij,bjk,bkm->bim', pitch_mat, yaw_mat, roll_mat)

    return rot_mat

def keypoint_transformation(kp_canonical, he, estimate_jacobian=False):
    kp = kp_canonical['value']    # (bs, k, 3)
    yaw, pitch, roll = he['yaw'], he['pitch'], he['roll']
    t, exp = he['t'], he['exp']
    
    yaw = headpose_pred_to_degree(yaw)
    pitch = headpose_pred_to_degree(pitch)
    roll = headpose_pred_to_degree(roll)

    rot_mat = get_rotation_matrix(yaw, pitch, roll)    # (bs, 3, 3)
    
    # keypoint rotation
    kp_rotated = torch.einsum('bmp,bkp->bkm', rot_mat, kp)
    # print(f't shape: {t.shape}')
    # print(f'kp shape: {kp.shape}')

    # keypoint translation

    t = t.unsqueeze_(1).repeat(1, kp.shape[1], 1)
    kp_t = kp_rotated + t


    kp_neutralized = kp_t

    # add expression deviation 
    exp = exp.view(exp.shape[0], -1, 3)
    kp_transformed = kp_t + exp

    if estimate_jacobian:
        jacobian = kp_canonical['jacobian']   # (bs, k ,3, 3)
        jacobian_transformed = torch.einsum('bmp,bkps->bkms', rot_mat, jacobian)
    else:
        jacobian_transformed = None

    return {'value': kp_transformed, 'jacobian': jacobian_transformed, 'neutralized': {'value': kp_neutralized, 'jacobian': jacobian_transformed}}

def keypoint_normalize(kp_canonical, he, estimate_jacobian=True):
    kp = kp_canonical['value']    # (bs, k, 3)
    yaw, pitch, roll = he['yaw'].detach(), he['pitch'].detach(), he['roll'].detach()
    t, exp = he['t'].detach(), he['exp']
    
    # keypoint translation
    t = t.unsqueeze(1).repeat(1, kp.shape[1], 1).detach()
    kp_t = kp - t

    yaw = headpose_pred_to_degree(yaw)
    pitch = headpose_pred_to_degree(pitch)
    roll = headpose_pred_to_degree(roll)

    rot_mat = get_rotation_matrix(-yaw, -pitch, -roll)    # (bs, 3, 3)


    # keypoint rotation
    kp_rotated = torch.einsum('bmp,bkp->bkm', rot_mat, kp_t)

    return {'value': kp_rotated, 'jacobian': None}

class NonStPCA():
    def __init__(self, dim_data, num_pc, update_freq=1, t=0.9):
        self.dim_data = dim_data
        self.num_pc = num_pc 
        self.update_freq = update_freq
        self.u = nn.init.orthogonal_(torch.empty(self.dim_data, self.num_pc))
        self.N = torch.randn(self.dim_data, self.num_pc)
        self.mu = torch.zeros(dim_data)
        self.s = torch.eye(self.num_pc)
        self.cnt = 0
        self.batch = []
        self.steps = 0

    def register(self, inp):
        # inp: d
        if self.cnt >= self.update_freq:
            if self.steps == 0:
                batch = torch.stack(self.batch, dim=0) # B x d
                mu = batch.mean(dim=0) # d
                self.mu = mu
                batch -= self.mu[None]
                u, s, v = torch.pca_lowrank(batch, q=self.num_pc)
                self.u = v
                self.s = torch.diag(s)
                self.batch = []
                self.steps += 1
        else:
            self.batch.append(inp)
            inp = inp - self.mu
            self.N += inp.unsqueeze(1) @ inp.unsqueeze(0) @ self.u / self.update_freq
            self.cnt += 1

    def get_state(self, device='cpu'):
        return self.mu.to(device), self.u.to(device), self.s.to(device)
    
    def load_state(self, mu, u, s):
        self.mu = mu.cpu()
        self.u = u.cpu()
        self.s = s.cpu()

class GeneratorFullModel(torch.nn.Module):
    """
    Merge all generator related updates into single model for better multi-gpu usage
    """

    def __init__(self, kp_extractor, he_estimator, generator, discriminator, train_params, estimate_jacobian=True):
        super(GeneratorFullModel, self).__init__()
        self.kp_extractor = kp_extractor
        self.he_estimator = he_estimator
        self.generator = generator
        self.discriminator = discriminator
        self.train_params = train_params
        self.scales = train_params['scales']
        self.disc_scales = self.discriminator.scales
        self.pyramid = ImagePyramide(self.scales, generator.image_channel)
        self.pyramid_cond = ImagePyramide(self.scales, generator.image_channel + 1)
        if torch.cuda.is_available():
            self.pyramid = self.pyramid.cuda()
            self.pyramid_cond = self.pyramid_cond.cuda()
            
        self.loss_weights = train_params['loss_weights']

        self.estimate_jacobian = estimate_jacobian

        if sum(self.loss_weights['perceptual']) != 0:
            self.vgg = Vgg19()
            if torch.cuda.is_available():
                self.vgg = self.vgg.cuda()

        if self.loss_weights['headpose'] != 0:
            self.hopenet = hopenet.Hopenet(models.resnet.Bottleneck, [3, 4, 6, 3], 66)
            print('Loading hopenet')
            hopenet_state_dict = torch.load(train_params['hopenet_snapshot'])
            self.hopenet.load_state_dict(hopenet_state_dict)
            if torch.cuda.is_available():
                self.hopenet = self.hopenet.cuda()
                self.hopenet.eval()


        self.sections = train_params['sections']
        self.split_ids = [sec[1] for sec in self.sections]
        self.pca_xs = []
        self.pca_es = []
        
        for i, sec in enumerate(self.sections):
            self.pca_xs.append(NonStPCA(3 * len(sec[0]), sec[1], update_freq=self.train_params['pca_update_freq']))
            self.pca_es.append(NonStPCA(3 * len(sec[0]), sec[1], update_freq=self.train_params['pca_update_freq']))

            self.register_buffer(f'mu_x_{i}', torch.Tensor(3 * len(sec[0])).cuda())
            self.register_buffer(f'u_x_{i}', torch.Tensor(3 * len(sec[0]), sec[1]).cuda())
            self.register_buffer(f's_x_{i}', torch.Tensor(sec[1], sec[1]).cuda())
            self.register_buffer(f'mu_e_{i}', torch.Tensor(3 * len(sec[0])).cuda())
            self.register_buffer(f'u_e_{i}', torch.Tensor(3 * len(sec[0]), sec[1]).cuda())
            self.register_buffer(f's_e_{i}', torch.Tensor(3 * len(sec[0]), sec[1]).cuda())
        
            self.register_buffer(f'sigma_err_{i}', (torch.eye(3 * len(sec[0])) * self.train_params['sigma_err']).cuda())

    def getattr(self, name):
        return getattr(self, name)
    
    def update_pc(self):
        for i, sec in enumerate(self.sections):
            mu_x, u_x, s_x = self.pca_xs[i].get_state(device='cuda')
            mu_e, u_e, s_e = self.pca_es[i].get_state(device='cuda')
            self.register_buffer(f'mu_x_{i}', mu_x)
            self.register_buffer(f'u_x_{i}', u_x)
            self.register_buffer(f's_x_{i}', s_x)
            self.register_buffer(f'mu_e_{i}', mu_e)
            self.register_buffer(f'u_e_{i}', u_e)
            self.register_buffer(f's_e_{i}', s_e)

    def load_pc(self):
        for i, sec in enumerate(self.sections):
            self.pca_xs[i].load_state(self.getattr(f'mu_x_{i}'), self.getattr(f'u_x_{i}'), self.getattr(f's_x_{i}'))
            self.pca_es[i].load_state(self.getattr(f'mu_e_{i}'), self.getattr(f'u_e_{i}'), self.getattr(f's_e_{i}'))

    # def regularize(self, kp_canonical_source, kp_canonical_driving, he_source, he_driving):
    #     loss = {}
    #     kp_source_normalized = keypoint_normalize(kp_canonical_source, he_source)['value'].flatten(1)   # B x N * 3 
    #     kp_driving_normalized = keypoint_normalize(kp_canonical_driving, he_driving)['value'].flatten(1) # B x N * 3
    #     # print(f'normalized kp shape: {kp_source_normalized.shape}')

    #     x = (kp_source_normalized + kp_driving_normalized) / 2
    #     e_source = kp_source_normalized - x


    #     k_e_grad = (he_source['exp'] + he_driving['exp']) / 2 # B x num_kp * 3
    #     k_e = k_e_grad.detach()

    #     loss['k_e'] = torch.norm(he_source['exp'] - he_driving['exp'], dim=1).mean() # 1
        
    #     x = x.detach().cpu()
    #     e = math.sqrt(2) * e_source.detach().cpu() / k_e.cpu().clamp(1e-1)

    #     for x_i in x:
    #         self.pca_x.register(x_i)
    #     for e_i in e:
    #         self.pca_e.register(e_i)

    #     self.update_pc()

    #     mu_x, u_x, s_x = self.pca_x.get_state(device='cuda')
    #     _, u_e, s_e = self.pca_e.get_state(device='cuda')


    #     print('regularizing')
    #     loss['regularizor'] = ((kp_source_normalized - mu_x[None]).unsqueeze(1) @ ((u_x @ (s_x ** 2) @ u_x.t())[None] + torch.diag_embed(k_e) @ u_e @ (s_e ** 2) @ u_e.t() @ torch.diag_embed(k_e) + self.sigma_err[None]).inverse() @ (kp_source_normalized - mu_x[None]).unsqueeze(2)).mean() # 1
    #     print(f'loss reg: {loss["regularizor"]}')

    #     u_e = torch.diag_embed(k_e_grad) @ u_e # B x 3 * num_kp x num_pc

    #     A = (torch.eye(self.train_params['num_pc']).cuda() + (u_x @ s_x).t() @ self.sigma_err.inverse() @ (u_x @ s_x))[None].repeat(len(x), 1, 1)  # B x num_pc x num_pc
    #     B = (u_e @ s_e).transpose(1, 2) @ self.sigma_err.inverse() @ (u_e @ s_e) # B x num_pc x num_pc
    #     C = (u_x @ s_x).t() @ self.sigma_err.inverse() @ (kp_source_normalized - mu_x[None]).unsqueeze(-1) # B x num_pc x num_pc
    #     A_ = (u_x @ s_x).t() @ self.sigma_err.inverse() @ (u_e @ s_e) # B x num_pc x num_pc
    #     B_ = torch.eye(self.train_params['num_pc']).cuda()[None] + (u_e @ s_e).transpose(1, 2) @ self.sigma_err.inverse() @ (u_e @ s_e)   # B x num_pc x num_pc
    #     C_ = (u_e @ s_e).transpose(1, 2) @ self.sigma_err.inverse() @ (kp_source_normalized - mu_x[None]).unsqueeze(-1)  # B x num_pc x num_pc

    #     M = torch.cat([torch.cat([A, B], dim=2), torch.cat([A_, B_], dim=2)], dim=1) # B x (2 * num_pc) x (2 * num_pc)
    #     N = torch.cat([C, C_], dim=1) # B x (2 * num_pc) x 1
    #     z_estimated = M.inverse() @ N # B x (2 * num_pc) x 1
    #     z_x_estimated, z_e_estimated = z_estimated.split(self.train_params['num_pc'], dim=1) # B x num_pc x 1
    #     kp_source_reg = mu_x.unsqueeze(0).unsqueeze(-1) + u_x @ s_x @ z_x_estimated + u_e @ s_e @ z_e_estimated # B x num_kp * 3 x 1
    #     kp_source_reg = kp_source_reg.squeeze(2).view(kp_source_reg.size(0), -1, 3) # B x num_kp x 3

    #     A = (torch.eye(self.train_params['num_pc']).cuda() + (u_x @ s_x).t() @ self.sigma_err.inverse() @ (u_x @ s_x))[None].repeat(len(x), 1, 1)  # B x num_pc x num_pc
    #     B = (u_e @ s_e).transpose(1, 2) @ self.sigma_err.inverse() @ (u_e @ s_e) # B x num_pc x num_pc
    #     C = (u_x @ s_x).t() @ self.sigma_err.inverse() @ (kp_driving_normalized - mu_x[None]).unsqueeze(-1) # B x num_pc x num_pc
    #     A_ = (u_x @ s_x).t() @ self.sigma_err.inverse() @ (u_e @ s_e) # B x num_pc x num_pc
    #     B_ = torch.eye(self.train_params['num_pc']).cuda()[None] + (u_e @ s_e).transpose(1, 2) @ self.sigma_err.inverse() @ (u_e @ s_e)   # B x num_pc x num_pc
    #     C_ = (u_e @ s_e).transpose(1, 2) @ self.sigma_err.inverse() @ (kp_driving_normalized - mu_x[None]).unsqueeze(-1)  # B x num_pc x num_pc

    #     M = torch.cat([torch.cat([A, B], dim=2), torch.cat([A_, B_], dim=2)], dim=1) # B x (2 * num_pc) x (2 * num_pc)
    #     N = torch.cat([C, C_], dim=1) # B x (2 * num_pc) x 1
    #     z_estimated = M.inverse() @ N # B x (2 * num_pc) x 1
    #     z_x_estimated, z_e_estimated = z_estimated.split(self.train_params['num_pc'], dim=1) # B x num_pc x 1
    #     kp_driving_reg = mu_x.unsqueeze(0).unsqueeze(-1) + u_x @ s_x @ z_x_estimated + u_e @ s_e @ z_e_estimated # B x num_kp * 3 x 1
    #     kp_driving_reg = kp_driving_reg.squeeze(2).view(kp_driving_reg.size(0), -1, 3) # B x num_kp x 3

    #     kp_source_reg = {'value': kp_source_reg, 'jacobian': None}
    #     kp_driving_reg = {'value': kp_driving_reg, 'jacobian': None}

    #     kp_source_reg = keypoint_transformation(kp_source_reg, he_source)
    #     kp_driving_reg = keypoint_transformation(kp_driving_reg, he_driving)
        
    #     res = {'kp_source': kp_source_reg, 'kp_driving': kp_driving_reg, 'loss': loss}

    #     return res

    def split_section(self, X):
        res = []
        for i, sec in enumerate(self.sections):
            res.append(X[:, sec[0]])
        return res
    
    def split_section_and_normalize(self, X):
        res = []
        secs = []
        for i, sec in enumerate(self.sections):
            sec_mean = X[:, sec[0]].mean(dim=1)
            res.append(X[:, sec[0]] - sec_mean.unsqueeze(1))
            secs.append(sec_mean)
        return res, secs
    
    def concat_section(self, sections):
        # sections[]: (num_sections) x B x -1 x 3
        return torch.cat(sections, dim=1)
    
    
    def register_keypoint(self, kp_source, kp_driving):
        X_source = kp_source['value']
        X_driving = kp_driving['value']

        X_source_splitted = self.split_section(X_source)
        X_driving_splitted = self.split_section(X_driving)
        
        for i, sec in enumerate(self.sections):
            x = (X_source_splitted[i] + X_driving_splitted[i]) / 2
            x = X_source_splitted[i]
            e = X_source_splitted[i] - x
            
            x = x.flatten(1).detach().cpu()
            e = math.sqrt(2) * e.flatten(1).detach().cpu()

            for x_i in x:
                self.pca_xs[i].register(x_i)
            for e_i in e:
                self.pca_es[i].register(e_i)
                
        self.update_pc()
        
    def calc_reg_loss(self, xs, es):
        # xs, es: (num_section) x N_i * 3
        loss_reg = 0
        
        for i, sec in enumerate(self.sections):
            x, e = xs[i], es[i]
            mu_x, u_x, s_x = self.pca_xs[i].get_state(device='cuda')
            mu_e, u_e, s_e = self.pca_es[i].get_state(device='cuda')
            sigma_err = self.getattr(f'sigma_err_{i}')
            loss_reg_x = ((x - mu_x[None]).unsqueeze(1) @ ((u_x @ (s_x ** 2) @ u_x.t())[None] + sigma_err[None]).inverse() @ (x - mu_x[None]).unsqueeze(2)).mean() # 1
            loss_reg_e = (e.unsqueeze(1) @ ((u_e @ (s_e ** 2) @ u_e.t())[None] + sigma_err[None]).inverse() @ e.unsqueeze(2)).mean() # 1
            
            loss_reg += (loss_reg_x + loss_reg_e)
        
        return loss_reg
    
    def calc_seg_loss(self, mask, heatmap):
        # heatmap: B x num_section x d x h x w
        # mask: B x (num_kp + 1) x d x h x w
        split_ids = [2]
        split_ids.extend(self.split_ids)
        seg_mask = torch.split(mask, split_ids, dim=1) # []: (num_section + 1) x B x -1 x d x h x w
        seg_mask = seg_mask[1:]
        seg_loss = 0
        for i in range(heatmap.size(1)):
            seg = seg_mask[i].sum(dim=1)
            seg_loss += ((1 - heatmap[:, [i]]) * seg).mean()
        
        return seg_loss

    def forward(self, x):
        # kp_canonical = self.kp_extractor(x['source'])     # {'value': value, 'jacobian': jacobian}   
        # kp_canonical_source = self.kp_extractor(x['source'])     # {'value': value, 'jacobian': jacobian}   
        # kp_canonical_driving = self.kp_extractor(x['driving'])     # {'value': value, 'jacobian': jacobian}   

        # he_source = self.he_estimator(x['source'])        # {'yaw': yaw, 'pitch': pitch, 'roll': roll, 't': t, 's_e': s_e}
        # he_driving = self.he_estimator(x['driving'])      # {'yaw': yaw, 'pitch': pitch, 'roll': roll, 't': t, 's_e': s_e}


        # driving_224 = x['hopenet_driving']
        # yaw_gt, pitch_gt, roll_gt = self.hopenet(driving_224)

        # reg = self.regularize(kp_canonical_source, kp_canonical_driving) # regularizor loss

        loss_values = {}
        
        bs = len(x['source_mesh']['value'])
        
        # x_reg = kp_canonical['value'].flatten(1)
        # for x_i in x_reg:
        #     self.pca_x.register(x_i.detach().cpu())
        # if self.pca_x.steps > 2:
        #     mu_x, u_x, s_x = self.pca_x.get_state(device='cuda')
        #     loss_reg = ((x_reg - mu_x[None]).unsqueeze(1) @ ((u_x @ (s_x ** 2) @ u_x.t())[None] + self.sigma_err[None]).inverse() @ (x_reg - mu_x[None]).unsqueeze(2)).mean() # 1
        #     loss['regularizor'] = self.loss_weights['regularizor'] * loss_reg
        #     self.mu_x, self.u_x, self.s_x = self.pca_x.get_state()
        # else:
        #     loss['regularizor'] = self.loss_weights['regularizor'] * torch.zeros(1).cuda().mean()

        # if self.pca_x.steps * self.pca_e.steps > 0:
        #     kp_source, kp_driving = reg['kp_source'], reg['kp_driving']
        #     loss = {k: self.loss_weights[k] * v for k, v in reg['loss'].items()}
        # else:
        #     kp_source, kp_driving = kp_canonical_source, kp_canonical_driving
        #     loss = {k: self.loss_weights[k] * torch.zeros(1).cuda() for k, v in reg['loss'].items()}


        # {'value': value, 'jacobian': jacobian}
        # kp_source = keypoint_transformation(kp_canonical, he_source, self.estimate_jacobian)
        # kp_driving = keypoint_transformation(kp_canonical, he_driving, self.estimate_jacobian)
        kp_source = x['source_mesh']
        kp_driving = x['driving_mesh']
        
        
        generated = self.generator(x['source'], kp_source=kp_source, kp_driving=kp_driving)
        src_section = self.concat_section(self.split_section(kp_source['value']))
        tgt_section = self.concat_section(self.split_section(kp_source['raw_value']))
        # print(f'src section: {src_section}')
        # print(f'drv section: {tgt_section}')
        generated['kp_source'] = {'value': src_section}
        generated['kp_driving'] = {'value': tgt_section}
        # seg_loss = self.calc_seg_loss(generated['mask'], generated['heatmap'])
        
        # loss_values['segmentation'] = self.loss_weights['segmentation'] * seg_loss

        pyramide_real = self.pyramid(x['driving'])
        pyramide_generated = self.pyramid(generated['prediction'])

        if sum(self.loss_weights['perceptual']) != 0:
            value_total = 0
            for scale in self.scales:
                x_vgg = self.vgg(pyramide_generated['prediction_' + str(scale)])
                y_vgg = self.vgg(pyramide_real['prediction_' + str(scale)])

                for i, weight in enumerate(self.loss_weights['perceptual']):
                    value = torch.abs(x_vgg[i] - y_vgg[i].detach()).mean()
                    value_total += self.loss_weights['perceptual'][i] * value
            loss_values['perceptual'] = value_total

        if self.loss_weights['generator_gan'] != 0:
            pyramide_real = self.pyramid_cond(torch.cat([kp_driving['mesh_img'].cuda(), x['driving']], dim=1))
            pyramide_generated = self.pyramid_cond(torch.cat([kp_source['mesh_img'].cuda(), generated['prediction']], dim=1))
            discriminator_maps_generated = self.discriminator(pyramide_generated)
            discriminator_maps_real = self.discriminator(pyramide_real)
            
            value_total = 0
            for scale in self.disc_scales:
                key = 'prediction_map_%s' % scale
                if self.train_params['gan_mode'] == 'hinge':
                    value = -torch.mean(discriminator_maps_generated[key])
                elif self.train_params['gan_mode'] == 'ls':
                    value = ((1 - discriminator_maps_generated[key]) ** 2).mean()
                else:
                    raise ValueError('Unexpected gan_mode {}'.format(self.train_params['gan_mode']))

                value_total += self.loss_weights['generator_gan'] * value
            loss_values['gen_gan'] = value_total

            if sum(self.loss_weights['feature_matching']) != 0:
                value_total = 0
                for scale in self.disc_scales:
                    key = 'feature_maps_%s' % scale
                    for i, (a, b) in enumerate(zip(discriminator_maps_real[key], discriminator_maps_generated[key])):
                        if self.loss_weights['feature_matching'][i] == 0:
                            continue
                        value = torch.abs(a - b).mean()
                        value_total += self.loss_weights['feature_matching'][i] * value
                    loss_values['feature_matching'] = value_total

        if (self.loss_weights['equivariance_value'] + self.loss_weights['equivariance_jacobian']) != 0:
            transform = Transform(x['driving'].shape[0], **self.train_params['transform_params'])
            transformed_frame = transform.transform_frame(x['driving'])

            transformed_he_driving = self.he_estimator(transformed_frame)

            transformed_kp = keypoint_normalize(kp_canonical, transformed_he_driving, self.estimate_jacobian)

            generated['transformed_frame'] = transformed_frame
            generated['transformed_kp'] = transformed_kp

            ## Value loss part
            if self.loss_weights['equivariance_value'] != 0:
                # project 3d -> 2d
                kp_driving_2d = kp_driving['value'][:, :, :2]
                transformed_kp_2d = transformed_kp['value'][:, :, :2]
                value = torch.abs(kp_driving_2d - transform.warp_coordinates(transformed_kp_2d)).mean()
                loss_values['equivariance_value'] = self.loss_weights['equivariance_value'] * value

            ## jacobian loss part
            if self.loss_weights['equivariance_jacobian'] != 0:
                # project 3d -> 2d
                transformed_kp_2d = transformed_kp['value'][:, :, :2]
                transformed_jacobian_2d = transformed_kp['jacobian'][:, :, :2, :2]
                jacobian_transformed = torch.matmul(transform.jacobian(transformed_kp_2d),
                                                    transformed_jacobian_2d)
                
                jacobian_2d = kp_driving['jacobian'][:, :, :2, :2]
                normed_driving = torch.inverse(jacobian_2d)
                normed_transformed = jacobian_transformed
                value = torch.matmul(normed_driving, normed_transformed)

                eye = torch.eye(2).view(1, 1, 2, 2).type(value.type())

                value = torch.abs(eye - value).mean()
                loss_values['equivariance_jacobian'] = self.loss_weights['equivariance_jacobian'] * value

        if self.loss_weights['keypoint'] != 0:
            # print(kp_driving['value'].shape)     # (bs, k, 3)
            value_total = 0
            for i in range(kp_driving['value'].shape[1]):
                for j in range(kp_driving['value'].shape[1]):
                    dist = F.pairwise_distance(kp_driving['value'][:, i, :], kp_driving['value'][:, j, :], p=2, keepdim=True) ** 2
                    dist = 0.1 - dist      # set Dt = 0.1
                    dd = torch.gt(dist, 0) 
                    value = (dist * dd).mean()
                    value_total += value

            kp_mean_depth = kp_driving['value'][:, :, -1].mean(-1)
            value_depth = torch.abs(kp_mean_depth - 0.33).mean()          # set Zt = 0.33

            value_total += value_depth
            loss_values['keypoint'] = self.loss_weights['keypoint'] * value_total

        if self.loss_weights['headpose'] != 0:
            # transform_hopenet =  transforms.Compose([
            #                                         transforms.Resize(size=(224, 224)),
            #                                         transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            #                                         transforms.ToTensor()])
            
            # print(f'driving image shape: {x["driving"][0].cpu().size()}')
            # print(f'driving image shape: {transforms.ToPILImage()(x["driving"][0].permute(1, 2, 0).cpu()).size}')
            # print(f'driving image: {transforms.ToPILImage()(x["driving"][0].permute(1, 2, 0).cpu())}')
            # driving_224 = transform_hopenet(x['driving'].cpu()).cuda()
            driving_224 = x['hopenet_driving']

            yaw_gt, pitch_gt, roll_gt = self.hopenet(driving_224)
            yaw_gt = headpose_pred_to_degree(yaw_gt)
            pitch_gt = headpose_pred_to_degree(pitch_gt)
            roll_gt = headpose_pred_to_degree(roll_gt)

            yaw, pitch, roll = he_driving['yaw'], he_driving['pitch'], he_driving['roll']
            yaw = headpose_pred_to_degree(yaw)
            pitch = headpose_pred_to_degree(pitch)
            roll = headpose_pred_to_degree(roll)

            value = torch.abs(yaw - yaw_gt).mean() + torch.abs(pitch - pitch_gt).mean() + torch.abs(roll - roll_gt).mean()
            loss_values['headpose'] = self.loss_weights['headpose'] * value

        if self.loss_weights['expression'] != 0:
            value = torch.norm(he_driving['exp'], p=1, dim=-1).mean()
            loss_values['expression'] = self.loss_weights['expression'] * value


        return loss_values, generated


class DiscriminatorFullModel(torch.nn.Module):
    """
    Merge all discriminator related updates into single model for better multi-gpu usage
    """

    def __init__(self, kp_extractor, generator, discriminator, train_params):
        super(DiscriminatorFullModel, self).__init__()
        self.kp_extractor = kp_extractor
        self.generator = generator
        self.discriminator = discriminator
        self.train_params = train_params
        self.scales = self.discriminator.scales
        self.pyramid = ImagePyramide(self.scales, generator.image_channel + 1)
        if torch.cuda.is_available():
            self.pyramid = self.pyramid.cuda()

        self.loss_weights = train_params['loss_weights']

        self.zero_tensor = None

    def get_zero_tensor(self, input):
        if self.zero_tensor is None:
            self.zero_tensor = torch.FloatTensor(1).fill_(0).cuda()
            self.zero_tensor.requires_grad_(False)
        return self.zero_tensor.expand_as(input)

    def forward(self, x, generated):
        # pyramide_real = self.pyramid(x['driving'])
        # pyramide_generated = self.pyramid(generated['prediction'].detach())

        pyramide_real = self.pyramid(torch.cat([x['driving_mesh']['mesh_img'].cuda(), x['driving']], dim=1))
        pyramide_generated = self.pyramid(torch.cat([x['source_mesh']['mesh_img'].cuda(), generated['prediction'].detach()], dim=1))
        
        discriminator_maps_generated = self.discriminator(pyramide_generated)
        discriminator_maps_real = self.discriminator(pyramide_real)

        loss_values = {}
        value_total = 0
        for scale in self.scales:
            key = 'prediction_map_%s' % scale
            if self.train_params['gan_mode'] == 'hinge':
                value = -torch.mean(torch.min(discriminator_maps_real[key]-1, self.get_zero_tensor(discriminator_maps_real[key]))) - torch.mean(torch.min(-discriminator_maps_generated[key]-1, self.get_zero_tensor(discriminator_maps_generated[key])))
            elif self.train_params['gan_mode'] == 'ls':
                value = ((1 - discriminator_maps_real[key]) ** 2 + discriminator_maps_generated[key] ** 2).mean()
            else:
                raise ValueError('Unexpected gan_mode {}'.format(self.train_params['gan_mode']))

            value_total += self.loss_weights['discriminator_gan'] * value
        loss_values['disc_gan'] = value_total

        return loss_values
