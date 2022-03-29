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

from logger import Logger

from frames_dataset import DatasetRepeater
from sync_batchnorm import DataParallelWithCallback

import torch.nn as nn

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
            
            
        # if self.cnt >= self.update_freq:
        #     batch = torch.stack(self.batch, dim=0) # B x d
        #     mu = batch.mean(dim=0) # d
        #     self.mu = (mu + self.mu) / 2
        #     batch -= self.mu[None]
        #     cov = (batch.unsqueeze(2) @ batch.unsqueeze(1)).mean(dim=0)
        #     q, _ = torch.qr(self.N)
        #     self.u = q
        #     self.s = torch.diag(torch.diag(q.t() @ cov @ q)).sqrt()
        #     self.cnt = 0
        #     self.steps += 1
        #     self.batch = []

    def get_state(self, device='cpu'):
        return self.mu.to(device), self.u.to(device), self.s.to(device)
    
    def load_state(self, mu, u, s):
        self.mu = mu.cpu()
        self.u = u.cpu()
        self.s = s.cpu()
        
        
class HeadModel(nn.Module):
    def __init__(self, train_params):
        super(HeadModel, self).__init__()
        self.train_params = train_params
        self.sections = train_params['sections']
        self.split_ids = [sec[1] for sec in self.sections]
        self.pca_xs = []
        self.pca_es = []
        
        
        self.scalers = nn.ModuleList()

        self.x_primes = [[] for _ in range(len(self.sections))]
        self.e_tildas = [[] for _ in range(len(self.sections))]
        self.e_primes = [[] for _ in range(len(self.sections))]
        
        for i, sec in enumerate(self.sections):
            self.register_buffer(f'mu_x_{i}', None)
            self.register_buffer(f'u_x_{i}', None)
            self.register_buffer(f's_x_{i}', None)
            self.register_buffer(f'u_e_{i}', None)
            self.register_buffer(f's_e_{i}', None)
        
            self.register_buffer(f'sigma_err_{i}', (torch.eye(3 * len(sec[0])) * self.train_params['sigma_err']).cuda())

            self.scalers.append(
                nn.Sequential(
                    nn.Linear(3 * len(sec[0]), 256),
                    nn.ReLU(),
                    nn.Linear(256, 3 * len(sec[0])),
                    nn.Sigmoid()
                )
            )
        
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
    
    def concat_section_and_denormalize(self, sections, means):
        return torch.cat([(sec + mean.unsqueeze(1)) for sec, mean in zip(sections, means)], dim=1)
    
    def get_mesh_bias(self):
        return torch.cat([self.getattr(f'mu_x_{i}') for i in range(len(self.sections))], dim=0).view(-1, 3) if self.getattr(f'mu_x_0') is not None else torch.cat([torch.zeros(3 * len(self.sections[i][0])).float().cuda() for i in range(len(self.sections))], dim=0).view(-1, 3)
    
    def register_keypoint(self, kp_source, kp_driving, k_e, coef_e_tilda=1, coef_e_prime=1):
        X_source = kp_source['value']
        X_driving = kp_driving['value']

        X_source_splitted, _ = self.split_section_and_normalize(X_source)
        X_driving_splitted, _ = self.split_section_and_normalize(X_driving)
        
        MAX_LEN = 1000
        if len(self.x_primes[0]) > MAX_LEN:
            return
        
        for i, sec in enumerate(self.sections):
            x = (X_source_splitted[i] + X_driving_splitted[i]) / 2
            e = X_source_splitted[i] - x
            
            x_prime = x.flatten(1).detach().cpu()
            e_tilda = (math.sqrt(2) * e).flatten(1).detach().cpu()
            e_prime = math.sqrt(2) * (e.flatten(1) / k_e[i].clamp(min=1e-3)).detach().cpu()
            
            for x_prime_i in x_prime:
                self.x_primes[i].append(x_prime_i)
                if len(self.x_primes[i]) > MAX_LEN:
                    del self.x_primes[i][0]
            for e_tilda_i in e_tilda: 
                self.e_tildas[i].append(e_tilda_i)
                if len(self.e_tildas[i]) > MAX_LEN:
                    del self.e_tildas[i][0]
            for e_prime_i in e_prime:
                self.e_primes[i].append(e_prime_i)
                if len(self.e_primes[i]) > MAX_LEN:
                    del self.e_primes[i][0]
                
        N = len(self.x_primes[0])
    
        if N > 100:
            for i, sec in enumerate(self.sections):
                x_primes = torch.stack(self.x_primes[i])
                e_tildas = torch.stack(self.e_tildas[i])
                e_primes = torch.stack(self.e_primes[i])
                
                mu_x = x_primes.mean(dim=0) # d
                x_primes = x_primes - mu_x[None]
                var_x_prime = (x_primes.t() @ x_primes) / (N - 1)
                var_e_tilda = (e_tildas.t() @ e_tildas) / (N - 1) * coef_e_tilda
                var_e_prime = (e_primes.t() @ e_primes) / (N - 1) * coef_e_prime
                
                var_x = var_x_prime - var_e_tilda / 2
                var_e = var_e_prime
                
                e, v = torch.eig(var_x, eigenvectors=True)
                # print(f'e, v shape: {e.shape}, {v.shape}')
                e = e[:, 0] # p0
                index = (-e).argsort()[:sec[1]]
                e = e[index]
                v = v[:, index]
                
                s = torch.diag(e).sqrt().cuda()
                u = v.cuda()
                
                self.register_buffer(f'mu_x_{i}', mu_x.cuda())
                self.register_buffer(f's_x_{i}', s)
                self.register_buffer(f'u_x_{i}', u)
                
                # print(f'mu_x_{i} shape: {mu_x}')
                # print(f's_x_{i} shape: {s}')
                # print(f'u_x_{i} shape: {u}')
                
                e, v = torch.eig(var_e, eigenvectors=True)
                e = e[:, 0] # p0
                index = (-e).argsort()[:sec[1]]
                e = e[index]
                v = v[:, index]
                
                s = torch.diag(e).sqrt().cuda()
                u = v.cuda()
            
                self.register_buffer(f's_e_{i}', s)
                self.register_buffer(f'u_e_{i}', u)
            
                # print(f's_e_{i} shape: {s}')
                # print(f'u_e_{i} shape: {u}')
                
                
                assert self.getattr('mu_x_0') is not None
                
            
            
            # for x_i in x:
            #     self.pca_xs[i].register(x_i)
            # for e_i in e:
            #     self.pca_es[i].register(e_i)
                
        # self.update_pc()

        # return X_source_splitted, X_driving_splitted
    
    def get_scale(self, sec, scaler):
        return 2 * scaler(sec)
    
    def extract_emotion_section(self, x, scaler, mu_x, u_x, s_x, u_e, s_e, sigma_err, num_pc):
        # x: B x N * 3   
        
        if mu_x is None:
            # k_e = x.abs().clamp(min=1e-3)
            k_e = self.get_scale(x, scaler) # B x n * 3
            kp_reg_x = torch.tensor(x)
            kp_reg_e = torch.zeros_like(x)
        
        else:
            # print('mu_x set')
            k_e = self.get_scale(x - mu_x[None], scaler) # B x n * 3
            # k_e = x.abs().clamp(min=1e-3)
            u_e = (k_e.unsqueeze(2) * u_e.unsqueeze(0)).view(len(k_e), *u_e.shape) # B x 3 * num_kp x num_pc
            
            A = (torch.eye(num_pc).cuda() + (u_x @ s_x).t() @ sigma_err.inverse() @ (u_x @ s_x))[None].repeat(len(x), 1, 1)  # B x num_pc x num_pc
            B = (u_e @ s_e).transpose(1, 2) @ sigma_err.inverse() @ (u_e @ s_e) # B x num_pc x num_pc
            C = (u_x @ s_x).t() @ sigma_err.inverse() @ (x - mu_x[None]).unsqueeze(-1) # B x num_pc x num_pc
            A_ = (u_x @ s_x).t() @ sigma_err.inverse() @ (u_e @ s_e) # B x num_pc x num_pc
            B_ = torch.eye(num_pc).cuda()[None] + (u_e @ s_e).transpose(1, 2) @ sigma_err.inverse() @ (u_e @ s_e)   # B x num_pc x num_pc
            C_ = (u_e @ s_e).transpose(1, 2) @ sigma_err.inverse() @ (x - mu_x[None]).unsqueeze(-1)  # B x num_pc x num_pc

            M = torch.cat([torch.cat([A, B], dim=2), torch.cat([A_, B_], dim=2)], dim=1) # B x (2 * num_pc) x (2 * num_pc)
            N = torch.cat([C, C_], dim=1) # B x (2 * num_pc) x 1
            z_estimated = M.inverse() @ N # B x (2 * num_pc) x 1
            z_x_estimated, z_e_estimated = z_estimated.split(num_pc, dim=1) # B x num_pc x 1
            kp_reg_x = (mu_x.unsqueeze(0).unsqueeze(-1) + u_x @ s_x @ z_x_estimated).squeeze(2) # B x num_kp * 3
            kp_reg_e = (u_e @ s_e @ z_e_estimated).squeeze(2) # B x num_kp * 3 
            
        return {'x': kp_reg_x, 'e': kp_reg_e, 'k_e': k_e}

    def extract_emotion(self, kp):
        kp_reg_xs = []
        kp_reg_es = []
        k_es = []
        
        secs, means = self.split_section_and_normalize(kp['value'])
        
        for i, sec in enumerate(secs):
            mu_x = self.getattr(f'mu_x_{i}')
            u_x = self.getattr(f'u_x_{i}')
            s_x = self.getattr(f's_x_{i}')
            u_e = self.getattr(f'u_e_{i}')
            s_e = self.getattr(f's_e_{i}')
            sigma_err = self.getattr(f'sigma_err_{i}')
            
            kp_reg = self.extract_emotion_section(sec.flatten(1), self.scalers[i], mu_x, u_x, s_x, u_e, s_e, sigma_err, self.sections[i][1])
            kp_reg_xs.append(kp_reg['x'])
            kp_reg_es.append(kp_reg['e'])
            k_es.append(kp_reg['k_e'])
        
        return {'x': kp_reg_xs, 'e': kp_reg_es, 'k_e': k_es, 'mean': means}
        
    def forward(self, x, eval=False):
        loss_values = {}
        generated = {}
        
        bs = len(x['source_mesh']['value'])
    
        kp_source = x['source_mesh']
        kp_driving = x['driving_mesh']
        
        source_res = self.extract_emotion(kp_source)    # {x, e, k_e}
        driving_res = self.extract_emotion(kp_driving)  # {x, e, k_e}
        
        source_recon = self.concat_section(driving_res['x']) + self.concat_section(source_res['e'])  # B x N * 3
        driving_recon = self.concat_section(source_res['x']) + self.concat_section(driving_res['e']) # B x N * 3
        recon = torch.cat([source_recon, driving_recon], dim=0) # 2 * B x N * 3
        
        k_e = torch.cat([self.concat_section(source_res['k_e']), self.concat_section(driving_res['k_e'])], dim=0)
        
        if not eval:
            self.register_keypoint(kp_source, kp_driving, source_res['k_e'], self.train_params['coef_e_tilda'], self.train_params['coef_e_prime'])
            
        source_raw, _ = self.split_section_and_normalize(kp_source['value'])
        driving_raw, _ = self.split_section_and_normalize(kp_driving['value'])
        
        source_raw = self.concat_section(source_raw).flatten(1)    # B x N * 3
        driving_raw = self.concat_section(driving_raw).flatten(1)   # B x N * 3
        raw = torch.cat([source_raw, driving_raw], dim=0)   # 2 * B x N * 3
        
        loss_values['recon'] = self.train_params['loss_weights']['recon'] * nn.MSELoss()(raw, recon)
        loss_values['k_e'] = self.train_params['loss_weights']['k_e'] * torch.norm(k_e - 1, dim=1).mean()
        
        generated['raw_source_X'] = self.concat_section(self.split_section(kp_source['value']))
        generated['raw_driving_X'] = self.concat_section(self.split_section(kp_driving['value']))
        generated['source_x'] = self.concat_section_and_denormalize([v.view(bs, -1, 3) for v in source_res['x']], source_res['mean'])
        generated['driving_x'] = self.concat_section_and_denormalize([v.view(bs, -1, 3) for v in driving_res['x']], driving_res['mean'])
        generated['source_e'] = self.concat_section(source_res['e']).view(bs, -1 ,3)
        generated['driving_e'] = self.concat_section(driving_res['e']).view(bs, -1, 3)
        # generated['mesh_bias'] = self.get_mesh_bias()[None].repeat(bs, 1, 1)
        generated['source_scale'] = self.concat_section(source_res['k_e'])
        generated['driving_scale'] = self.concat_section(driving_res['k_e'])
        
        return loss_values, generated

        
def main(config, model, log_dir, dataset, eval_dataset=None):
    train_params = config['train_params']
    
    model = DataParallelWithCallback(model)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=train_params['lr_headmodel'], betas=(0.5, 0.999))
    
    start_epoch = 0

    dataloader = DataLoader(dataset, batch_size=train_params['batch_size'], shuffle=True, num_workers=0, drop_last=True)
    if eval_dataset is not None:
        eval_dataloader = DataLoader(eval_dataset, batch_size=10, shuffle=False, num_workers=0, drop_last=False)
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
    parser.add_argument("--config", default="config/vox-256.yaml", help="path to config")
    parser.add_argument("--mode", default="train", choices=["train",])
    parser.add_argument("--gen", default="spade", choices=["original", "spade"])
    parser.add_argument("--log_dir", default='log_headmodel', help="path to log into")
    parser.add_argument("--checkpoint", default=None, help="path to checkpoint to restore")
    
    parser.add_argument("--device_ids", default="0", type=lambda x: list(map(int, x.split(','))),
                        help="Names of the devices comma separated.")
    parser.add_argument("--verbose", dest="verbose", action="store_true", help="Print model architecture")
    parser.set_defaults(verbose=False)
    parser.add_argument("--coef_e_tilda", type=float, default=1.0)
    parser.add_argument("--coef_e_prime", type=float, default=1.0)
    
    opt = parser.parse_args()
    with open(opt.config) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
        
    config['train_params']['sections'] = config['model_params']['common_params']['sections']
    config['train_params']['coef_e_tilda'] = opt.coef_e_tilda
    config['train_params']['coef_e_prime'] = opt.coef_e_prime
    
    dataset = FramesDataset(is_train=(opt.mode == 'train'), **config['dataset_params'])
    eval_dataset = FramesDataset(is_train=False, **config['dataset_params'])
    model = HeadModel(config['train_params']).cuda()
    
    if opt.checkpoint is not None:
        log_dir = os.path.join(*os.path.split(opt.checkpoint)[:-1])
    else:
        log_dir = os.path.join(opt.log_dir, os.path.basename(opt.config).split('.')[0])
        log_dir += ' {} {}'.format(opt.coef_e_tilda, opt.coef_e_prime)
        log_dir += ' ' + strftime("%d_%m_%y_%H.%M.%S", gmtime())
        
    print(f'Dataset Size: {len(dataset)}')
    print(f'Eval Dataset Size: {len(eval_dataset)}')    
    
    main(config, model, log_dir, dataset, eval_dataset=eval_dataset)

