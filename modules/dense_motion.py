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
        self.hourglass = Hourglass(block_expansion=block_expansion, in_features=(num_kp+1)*compress + len(sections), max_features=max_features, num_blocks=num_blocks)
        
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
        self.emotion_estimators = nn.ModuleList()
        self.kp_extractors = nn.ModuleList()
        
        for i, sec in enumerate(self.sections):
            self.emotion_estimators.append(nn.Sequential(
                nn.Linear(3 * len(sec[0]), 128),
                nn.ReLU(),
                nn.Linear(128, 3 * len(sec[0]))
            ))
            self.kp_extractors.append(nn.Sequential(
                nn.Linear(3 * len(sec[0]), 128),
                nn.ReLU(),
                nn.Linear(128, 3 * sec[1])
            ))
            
    def split_section(self, X):
        res = []
        for i, sec in enumerate(self.sections):
            res.append(X[:, sec[0]])
        return res
    
        
    def create_sparse_motions(self, feature, kp_driving, kp_source):
        bs, _, d, h, w = feature.shape
        identity_grid = make_coordinate_grid((d, h, w), type=kp_source['kp'].type())
        identity_grid = identity_grid.view(d, h, w, 3)
        
        delta = kp_source['kp'] - kp_driving['kp']     # B x (num_kp - 1) x 3
        delta = torch.cat([torch.zeros_like(delta[:, [0]]), delta], dim=1)  # B x num_kp x 3
        frontalized_grid = torch.einsum('dhwp,bnp->bdhwn', identity_grid, kp_driving['R'])
        scaled_grid = torch.einsum('bdhwn,b->bdhwn', frontalized_grid, kp_driving['c'])
        normalized_grid = scaled_grid + kp_driving['t'].squeeze(-1).unsqueeze(1).unsqueeze(2).unsqueeze(3) # B x D x H x W x 3
        transformed_grid = normalized_grid.unsqueeze(1) + delta.unsqueeze(2).unsqueeze(3).unsqueeze(4) # B x K x D x H x W x 3
        denormalized_grid = transformed_grid - kp_source['t'].squeeze(-1).unsqueeze(1).unsqueeze(2).unsqueeze(3).unsqueeze(4)
        descaled_grid = torch.einsum('bkdhwn,b->bkdhwn', denormalized_grid, 1 / kp_source['c'])
        # print(f'type: {kp_source["R"]}')
        # print(f'type: {descaled_grid}')
        defrontalized_grid = torch.einsum('bkdhwp,bnp->bkdhwn', descaled_grid.float(), kp_source['R'].inverse())
        
        driving_to_source = defrontalized_grid
        
        # coordinate_grid = identity_grid - kp_driving['value'].view(bs, self.num_kp, 1, 1, 1, 3)
        
        # k = coordinate_grid.shape[1]
        
        # # if 'jacobian' in kp_driving:
        # if 'jacobian' in kp_driving and kp_driving['jacobian'] is not None:
        #     jacobian = torch.matmul(kp_source['jacobian'], torch.inverse(kp_driving['jacobian']))
        #     jacobian = jacobian.unsqueeze(-3).unsqueeze(-3).unsqueeze(-3)
        #     jacobian = jacobian.repeat(1, 1, d, h, w, 1, 1)
        #     coordinate_grid = torch.matmul(jacobian, coordinate_grid.unsqueeze(-1))
        #     coordinate_grid = coordinate_grid.squeeze(-1)
        # '''
        # if 'rot' in kp_driving:
        #     rot_s = kp_source['rot']
        #     rot_d = kp_driving['rot']
        #     rot = torch.einsum('bij, bjk->bki', rot_s, torch.inverse(rot_d))
        #     rot = rot.unsqueeze(-3).unsqueeze(-3).unsqueeze(-3).unsqueeze(-3)
        #     rot = rot.repeat(1, k, d, h, w, 1, 1)
        #     # print(rot.shape)
        #     coordinate_grid = torch.matmul(rot, coordinate_grid.unsqueeze(-1))
        #     coordinate_grid = coordinate_grid.squeeze(-1)
        #     # print(coordinate_grid.shape)
        # '''
        # driving_to_source = coordinate_grid + kp_source['value'].view(bs, self.num_kp, 1, 1, 1, 3)    # (bs, num_kp, d, h, w, 3)

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

    def create_heatmap_representations(self, feature, kp_driving):
        spatial_size = feature.shape[3:]
        # print(f'section shape: {kp_driving["section"].shape}')
        heatmap = kp2gaussian({'value': kp_driving['section']}, spatial_size=spatial_size, kp_variance=0.01)
        # adding background feature
        return heatmap

    def extract_emotion(self, kp):
        xs = []
        es = []
        kps = []
        scs = []
        
        for i, sec in enumerate(self.sections):
            # print(f'kp value reporting: {kp["value"]}')
            # print("12222222")
            kp_splitted = kp['value'][:, sec[0]] # B x N x 3
            bs, n, _ = kp_splitted.shape
            sc = kp_splitted.mean(dim=1) # B x 3
            sc = kp['R'].inverse() @ (sc.unsqueeze(-1) - kp['t']) / kp['c'].unsqueeze(1).unsqueeze(2)
            scs.append(sc.squeeze(-1))
            kp_unbiased = kp_splitted - kp['mesh_bias'][i].view(1, n, 3)
            e = self.emotion_estimators[i](kp_unbiased.flatten(1)) # B x N * 3
            x = kp_splitted.flatten(1) - e # B x N * 3
            es.append(e)
            xs.append(x)
            kps.append(self.kp_extractors[i](e).view(bs, sec[1], 3)) # B x k x 3
            
        kp['section'] = torch.stack(scs, dim=1) # B x num_section x 3
        kp['kp'] = torch.cat(kps, dim=1)    # B x (num_kp - 1) x 3
        kp['x'] = xs
        kp['e'] = es
        
    def forward(self, feature, kp_driving, kp_source):
        bs, _, d, h, w = feature.shape

        feature = self.compress(feature)
        feature = self.norm(feature)
        feature = F.relu(feature)

        out_dict = dict()
        
        # add 'emotion', 'section' item for each kp
        self.extract_emotion(kp_driving)
        self.extract_emotion(kp_source)
        
        out_dict['x_source'] = kp_source['x']
        out_dict['e_source'] = kp_source['e']
        

        sparse_motion = self.create_sparse_motions(feature, kp_driving, kp_source)
        deformed_feature = self.create_deformed_feature(feature, sparse_motion)

        heatmap = self.create_heatmap_representations(deformed_feature, kp_driving)
        out_dict['heatmap'] = heatmap
        
        input = torch.cat([heatmap, deformed_feature.view(bs, -1, d, h, w)], dim=1)

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
