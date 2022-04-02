from torch import nn
import torch.nn.functional as F
import torch
from modules.util import Hourglass, make_coordinate_grid, kp2gaussian

from sync_batchnorm import SynchronizedBatchNorm3d as BatchNorm3d


class DenseMotionNetwork(nn.Module):
    """
    Module that predicting a dense motion from sparse motion representation given by kp_source and kp_driving
    """

    def __init__(self, block_expansion, num_blocks, sections, max_features, num_kp, feature_channel, reshape_depth, compress,
                 estimate_occlusion_map=False):
        super(DenseMotionNetwork, self).__init__()
        # self.hourglass = Hourglass(block_expansion=block_expansion, in_features=(num_kp+1)*(feature_channel+1), max_features=max_features, num_blocks=num_blocks)
        self.hourglass = Hourglass(block_expansion=block_expansion, in_features=(num_kp+1)*(compress+1), max_features=max_features, num_blocks=num_blocks)
        
        self.mask = nn.Conv3d(self.hourglass.out_filters, num_kp + 1, kernel_size=7, padding=3)

        self.compress = nn.Conv3d(feature_channel, compress, kernel_size=1)
        self.norm = BatchNorm3d(compress, affine=True)

        if estimate_occlusion_map:
            # self.occlusion = nn.Conv2d(reshape_channel*reshape_depth, 1, kernel_size=7, padding=3)
            self.occlusion = nn.Conv2d(self.hourglass.out_filters*reshape_depth, 1, kernel_size=7, padding=3)
        else:
            self.occlusion = None

        self.num_kp = num_kp
        self.sections = sections

        self.split_ids = [sec[1] for sec in self.sections]

            
    def split_section(self, X):
        res = []
        for i, sec in enumerate(self.sections):
            res.append(X[:, sec[0]])
        return res
    
        
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
        

    def forward(self, feature, kp_driving, kp_source):
        bs, _, d, h, w = feature.shape

        feature = self.compress(feature)
        feature = self.norm(feature)
        feature = F.relu(feature)

        out_dict = dict()
        
        rotation_kps = self.extract_rotation_keypoints(kp_source, kp_driving)
        kp_source['kp'] = rotation_kps['src']
        kp_driving['kp'] = rotation_kps['drv']
        
        out_dict['kp_source'] = {'value': kp_source['kp']}
        out_dict['kp_driving'] = {'value': kp_driving['kp']}
        
        sparse_motion = self.create_sparse_motions(feature, kp_driving, kp_source)
        deformed_feature = self.create_deformed_feature(feature, sparse_motion)
        
        heatmap = self.create_heatmap_representations(deformed_feature, kp_driving, kp_source)
        out_dict['heatmap'] = heatmap


        input = torch.cat([heatmap, deformed_feature], dim=2)
        input = input.view(bs, -1, d, h, w)

        # input = deformed_feature.view(bs, -1, d, h, w)      # (bs, num_kp+1 * c, d, h, w)

        prediction = self.hourglass(input)

        mask = self.mask(prediction)
        mask = F.softmax(mask, dim=1)
        out_dict['mask'] = mask
        mask = mask.unsqueeze(2)                                   # (bs, num_kp+1, 1, d, h, w)
        sparse_motion = sparse_motion.permute(0, 1, 5, 2, 3, 4)    # (bs, num_kp+1, 3, d, h, w)
        deformation = (sparse_motion * mask).sum(dim=1)            # (bs, 3, d, h, w)
        deformation = deformation.permute(0, 2, 3, 4, 1)           # (bs, d, h, w, 3)

        out_dict['deformation'] = deformation

        if self.occlusion:
            bs, c, d, h, w = prediction.shape
            prediction = prediction.view(bs, -1, h, w)
            occlusion_map = torch.sigmoid(self.occlusion(prediction))
            out_dict['occlusion_map'] = occlusion_map

        return out_dict
