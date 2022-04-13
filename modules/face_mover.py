from torch import nn
import torch.nn.functional as F
import torch
from modules.util import Hourglass, make_coordinate_grid, make_coordinate_grid_2d, kp2gaussian, AntiAliasInterpolation2d

from sync_batchnorm import SynchronizedBatchNorm3d as BatchNorm3d
from sync_batchnorm import SynchronizedBatchNorm2d as BatchNorm2d



class FaceMover(nn.Module):
    """
    Module that predicting a dense motion from sparse motion representation given by kp_source and kp_driving
    """

    def __init__(self, block_expansion, num_blocks, sections, max_features, num_kp, feature_channel, reshape_depth, compress,
                 estimate_occlusion_map=False):
        super(FaceMover, self).__init__()
        # self.hourglass = Hourglass(block_expansion=block_expansion, in_features=(num_kp+1)*(feature_channel+1), max_features=max_features, num_blocks=num_blocks)
        self.hourglass = Hourglass(block_expansion=block_expansion, in_features=(3 + 1), max_features=max_features // 2, num_blocks=num_blocks, is_2D=True)
        
        self.masks = nn.ModuleList()
        
        for sec in sections:
            self.masks.append(nn.Sequential(
                nn.Conv2d(self.hourglass.out_filters, sec[1], kernel_size=3, padding=1),
                nn.Sigmoid()))
        
        self.pyramid = AntiAliasInterpolation2d(sum([sec[1] for sec in sections]), 0.25)
        self.num_kp = num_kp
        self.sections = sections

        self.split_ids = [sec[1] for sec in self.sections]

        self.prior_extractors = nn.ModuleList()
        for i, sec in enumerate(self.sections):
            self.prior_extractors.append(nn.Sequential(
                nn.Linear(3 * len(sec[0]), 128),
                nn.ReLU(),
                nn.Linear(128, 3 * sec[1]),
                nn.Tanh()
            ))
        

    def extract_coef(self, kp, src_image):
        bs = kp['intermediate_mesh_img_sec'][0].shape[0]
        section_images = torch.cat([torch.cat([src_image, sec], dim=1) for sec in kp['intermediate_mesh_img_sec']], dim=0)
        features = self.hourglass(section_images).split(bs, dim=0)
        coefs = []
        for i, f in enumerate(features):
            coef = self.masks[i](nn.ReLU()(f))
            coefs.append(coef)
        coefs = torch.cat(coefs, dim=1)
        
        return coefs
    
    def extract_delta(self, kp_source, kp_driving):
        prior_driving, means = self.extract_prior(kp_driving)
        prior_source, _ = self.extract_prior(kp_source)
        # prior_source, _ = self.extract_prior(kp_source, MEANS=means)
        rotated_prior_driving = torch.einsum('bij,bnj->bni', kp_source['R'].inverse() / kp_source['c'][:, None, None], prior_driving - kp_source['t'][:, None, :, 0])
        rotated_prior_source = torch.einsum('bij,bnj->bni', kp_source['R'].inverse() / kp_source['c'][:, None, None], prior_source - kp_source['t'][:, None, :, 0])
        delta = rotated_prior_source - rotated_prior_driving
        delta = delta[:, :, :2]
        
        return delta, means
        
    def extract_prior(self, kp, use_intermediate=False, MEANS=None):
        mesh = kp['value'] if not use_intermediate else kp['intermediate_value'] # B x N x 3
        bs = len(mesh)
        if MEANS is None:
            secs, ms = self.split_section_and_normalize(mesh) # (num_sections) x B x n x 3, (num_sections) x B x 3
            means = []
        else:
            secs = self.split_section(mesh)
            ms = MEANS.split([sec[1] for sec in self.sections], dim=1)
            
        priors = []
        
        for i, sec in enumerate(secs):
            if MEANS is not None:
                sec = sec - ms[i][:, [0]]
            prior = self.prior_extractors[i](sec.flatten(1)).view(bs, -1, 3) # B x num_prior x 2
            priors.append(prior)
            if MEANS is None:
                means.append(ms[i].unsqueeze(1).repeat(1, prior.shape[1], 1))   # B x num_priors x 3
        priors = torch.cat(priors, dim=1)
        if MEANS is None:
            means = torch.cat(means, dim=1)
        else:
            means = MEANS
            
        return priors, means
    
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
        
    
    def create_sparse_motions(self, feature, kp_driving, kp_source):
        bs, _, d, h, w = feature.shape
        identity_grid = make_coordinate_grid((d, h, w), type=kp_source['kp'].type())
        identity_grid = identity_grid.view(1, 1, d, h, w, 3)
        coordinate_grid = identity_grid - kp_driving['kp'].view(bs, self.num_kp, 1, 1, 1, 3)
        
        jacobian = (kp_driving['c'] / kp_source['c']).unsqueeze(1).unsqueeze(2) * kp_driving['R'] @ kp_source['R'].inverse() # B x 3 x 3
        coordinate_grid = torch.einsum('bij,bklmnj->bklmni', jacobian, coordinate_grid)
        
        k = coordinate_grid.shape[1]
        
        # if 'jacobian' in kp_driving:
        if 'jacobian' in kp_driving and kp_driving['jacobian'] is not None:
            jacobian = torch.matmul(kp_source['jacobian'], torch.inverse(kp_driving['jacobian']))
            jacobian = jacobian.unsqueeze(-3).unsqueeze(-3).unsqueeze(-3)
            jacobian = jacobian.repeat(1, 1, d, h, w, 1, 1)
            coordinate_grid = torch.matmul(jacobian, coordinate_grid.unsqueeze(-1))
            coordinate_grid = coordinate_grid.squeeze(-1)
        '''
        if 'rot' in kp_driving:
            rot_s = kp_source['rot']
            rot_d = kp_driving['rot']
            rot = torch.einsum('bij, bjk->bki', rot_s, torch.inverse(rot_d))
            rot = rot.unsqueeze(-3).unsqueeze(-3).unsqueeze(-3).unsqueeze(-3)
            rot = rot.repeat(1, k, d, h, w, 1, 1)
            # print(rot.shape)
            coordinate_grid = torch.matmul(rot, coordinate_grid.unsqueeze(-1))
            coordinate_grid = coordinate_grid.squeeze(-1)
            # print(coordinate_grid.shape)
        '''
        driving_to_source = coordinate_grid + kp_source['kp'].view(bs, self.num_kp, 1, 1, 1, 3)    # (bs, num_kp, d, h, w, 3)

        #adding background feature
        identity_grid = identity_grid.repeat(bs, 1, 1, 1, 1, 1)
        sparse_motions = torch.cat([identity_grid, driving_to_source], dim=1)
        
        # sparse_motions = driving_to_source

        return sparse_motions

    def create_deformed_feature(self, feature, sparse_motions):
        bs, _, d, h, w = feature.shape
        feature_repeat = feature.unsqueeze(1).unsqueeze(1).repeat(1, self.num_kp+1, 1, 1, 1, 1, 1)      # (bs, num_kp+1, 1, c, d, h, w)
        feature_repeat = feature_repeat.view(bs * (self.num_kp+1), -1, d, h, w)                         # (bs*(num_kp+1), c, d, h, w)
        sparse_motions = sparse_motions.view((bs * (self.num_kp+1), d, h, w, -1))                       # (bs*(num_kp+1), d, h, w, 3)
        sparse_deformed = F.grid_sample(feature_repeat, sparse_motions)
        sparse_deformed = sparse_deformed.view((bs, self.num_kp+1, -1, d, h, w))                        # (bs, num_kp+1, c, d, h, w)
        return sparse_deformed


    def create_heatmap_representations(self, feature, kp_driving, kp_source):
        spatial_size = feature.shape[3:]
        gaussian_driving = kp2gaussian(kp_driving, spatial_size=spatial_size, kp_variance=0.01)
        gaussian_source = kp2gaussian(kp_source, spatial_size=spatial_size, kp_variance=0.01)
        heatmap = gaussian_driving - gaussian_source

        # adding background feature
        zeros = torch.zeros(heatmap.shape[0], 1, spatial_size[0], spatial_size[1], spatial_size[2]).type(heatmap.type())
        heatmap = torch.cat([zeros, heatmap], dim=1)
        heatmap = heatmap.unsqueeze(2)         # (bs, num_kp+1, 1, d, h, w)
        return heatmap
        
    def extract_rotation_keypoints(self, kp_source, kp_driving):
        mesh = kp_source['value'] # B x N x 3
        max_x, min_x = mesh[:, :, 0].max(dim=1)[0], mesh[:, :, 0].min(dim=1)[0]   # B
        max_y, min_y = mesh[:, :, 1].max(dim=1)[0], mesh[:, :, 1].min(dim=1)[0]   # B
        max_z = mesh[:, :, 2].max(dim=1)[0]    # B
        min_z = - max_z # B
        x_coords = torch.stack([min_x, (max_x + min_x) / 2, max_x], dim=1) # B x 3
        y_coords = torch.stack([min_y, (max_y + min_y) / 2, max_y], dim=1) # B x 3
        z_coords = torch.stack([min_z, (max_z + min_z) / 2, max_z], dim=1) # B x 3
        coords = []
        for x_coord, y_coord, z_coord in zip(x_coords, y_coords, z_coords):
            coord = torch.cartesian_prod(x_coord, y_coord, z_coord) # 3 * 3 * 3 x 3
            coords.append(coord)
        coords = torch.stack(coords, dim=0) # B x N x 3
        
        # coords scaling
        center = coords[:, 14].unsqueeze(1) # B x 1 x 3
        coords = center + 0.5 * (coords - center)
        
        # coords pooling
        coords = coords.view(-1, 3, 3, 3, 3)
        coords = coords[:, [0, 2]][:, :, [0, 2]][:, :, :, [0, 2]] # B x 8 x 3
        coords = coords.view(len(coords), -1, 3)
        
        coords = center
        
        coords_src = coords - kp_source['t'].unsqueeze(1).squeeze(3) # B x N x 3
        coords_src = torch.einsum('bij,bnj->bni', kp_source['R'].inverse() / kp_source['c'].unsqueeze(1).unsqueeze(2), coords_src) # B x N x 3
        
        coords_drv = coords - kp_driving['t'].unsqueeze(1).squeeze(3) # B x N x 3
        coords_drv = torch.einsum('bij,bnj->bni', kp_driving['R'].inverse() / kp_driving['c'].unsqueeze(1).unsqueeze(2), coords_drv) # B x N x 3

        return {'src': coords_src, 'drv': coords_drv}         
        

    def forward(self, kp_source, kp_driving, feature, src_image):
        # driving_mesh_img: B x 1 x H x W
        
        ### FIX: 3D feature 2D Warping !!!!!!!!! ####
        
        bs, _, h, w = feature.shape
        
        out = dict()
        
        coefs = self.extract_coef(kp_driving, src_image)
        out['coefs'] = coefs # B x num_priors x 2

        # coefs = self.pyramid(coefs)
        
        # driving_priors, means = self.extract_prior(kp_driving, use_intermediate=True)
        # source_priors, _ = self.extract_prior(kp_source)
        
        # out['driving_priors'] = driving_priors
        # out['source_priors'] = source_priors
        
        # out['means'] = means # B x num_sections x 3
        # out['coefs'] = coefs # B x num_priors x 2
        
        # delta = source_priors - driving_priors # B x num_priors x  2
        
        delta, means = self.extract_delta(kp_source, kp_driving)
        out['means'] = means
        
        move = torch.einsum('bphw,bpn->bnhw', coefs, delta)   # B x 2 x H x W
        move = move.permute(0, 2, 3, 1) # B x H x W x 2
        # move = torch.cat([move, torch.zeros_like(move[:, :, :, :1])], dim=3) # B x H x W x 2
        
    
        identity_grid = make_coordinate_grid_2d((h, w), type=kp_source['value'].type()).unsqueeze(0)
        # print(f'move shape: {move.shape}')
        # print(f'indentity shape: {identity_grid.shape}')
        move = identity_grid + move
        out['move'] = move
        moved_feature = nn.functional.grid_sample(feature, move)
        
        # print(f'moved feature shape: {moved_feature.shape}')
    
        out['moved_feature'] = moved_feature
        
        return out

