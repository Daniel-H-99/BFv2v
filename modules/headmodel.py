import math
import os, sys
import yaml
from argparse import ArgumentParser
from time import gmtime, strftime
from shutil import copy
from utils.util import block_diagonal_batch
import torch
import warnings


from tqdm import trange
from tqdm import tqdm
import torch

import torch.nn as nn
import scipy.stats as stats

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
        self.sections = train_params['headmodel_sections']
        self.split_ids = [sec[1] for sec in self.sections]
        self.pca_xs = []
        self.pca_es = []
        
        
        self.scalers = nn.ModuleList()

        self.x_primes = [[] for _ in range(len(self.sections))]
        self.e_tildas = [[] for _ in range(len(self.sections))]
        self.e_primes = [[] for _ in range(len(self.sections))]
        self.s_err_square = self.train_params['sigma_err']
        
        for i, sec in enumerate(self.sections):
            self.register_buffer(f'mu_x_{i}', torch.zeros(3 * len(sec[0])).cuda())
            self.register_buffer(f'u_x_{i}', torch.zeros(3 * len(sec[0]), sec[1]).cuda())
            self.register_buffer(f's_x_{i}', torch.eye(sec[1]).cuda())
            self.register_buffer(f'u_e_{i}', torch.zeros(3 * len(sec[0]), sec[1]).cuda())
            self.register_buffer(f's_e_{i}', torch.eye(sec[1]).cuda())
        
            self.register_buffer(f'sigma_err_{i}', (torch.eye(3 * len(sec[0])) * self.s_err_square).cuda())

            self.scalers.append(
                nn.Sequential(
                    nn.Linear(3 * len(sec[0]), 256),
                    nn.ReLU(),
                    # nn.Linear(256, 256),
                    # nn.ReLU(),
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

        # X_source_splitted, _ = self.split_section_and_normalize(X_source)
        # X_driving_splitted, _ = self.split_section_and_normalize(X_driving)

        X_source_splitted = self.split_section(X_source)
        X_driving_splitted = self.split_section(X_driving) 
        
        MAX_LEN = 1000
        if len(self.x_primes[0]) > MAX_LEN:
            return
        
        for i, sec in enumerate(self.sections):
            x = (X_source_splitted[i] + X_driving_splitted[i]) / 2
            e = X_source_splitted[i] - x
            
            x_prime = x.flatten(1).detach().cpu()
            e_tilda = (math.sqrt(2) * e).flatten(1).detach().cpu()
            e_prime = math.sqrt(2) * e.flatten(1).detach().cpu()
            
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
    
    def extract_emotion_section(self, x, scaler, mu_x, u_x, s_x, u_e, s_e, sigma_err):
        # x: B x N * 3   
        
        num_pc = len(s_x)

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
            B = (u_x @ s_x).t() @ sigma_err.inverse() @ (u_e @ s_e) # B x num_pc x num_pc
            C = (u_x @ s_x).t() @ sigma_err.inverse() @ (x - mu_x[None]).unsqueeze(-1) # B x num_pc x num_pc
            A_ = (u_e @ s_e).transpose(1, 2) @ sigma_err.inverse() @ (u_x @ s_x) # B x num_pc x num_pc
            B_ = torch.eye(num_pc).cuda()[None] + (u_e @ s_e).transpose(1, 2) @ sigma_err.inverse() @ (u_e @ s_e)   # B x num_pc x num_pc
            C_ = (u_e @ s_e).transpose(1, 2) @ sigma_err.inverse() @ (x - mu_x[None]).unsqueeze(-1)  # B x num_pc x 1

            M = torch.cat([torch.cat([A, B], dim=2), torch.cat([A_, B_], dim=2)], dim=1) # B x (2 * num_pc) x (2 * num_pc)
            N = torch.cat([C, C_], dim=1) # B x (2 * num_pc) x 1
            z_estimated = M.inverse() @ N # B x (2 * num_pc) x 1
            z_x_estimated, z_e_estimated = z_estimated.split(num_pc, dim=1) # B x num_pc x 1
            kp_reg_x = (mu_x.unsqueeze(0).unsqueeze(-1) + u_x @ s_x @ z_x_estimated).squeeze(2) # B x num_kp * 3
            kp_reg_e = (u_e @ s_e @ z_e_estimated).squeeze(2) # B x num_kp * 3 
            
            # grad_x = A @ z_x_estimated + B @ z_e_estimated - C
            # grad_e = A_ @ z_x_estimated + B_ @ z_e_estimated - C_
            # tmp = u_x @ s_x @ z_x_estimated
            # print(f'recon x shape: {tmp.shape}')
            # tmp = u_e @ s_e @ z_e_estimated
            # print(f'recon e shape: {tmp.shape}')
            # recon_err = ((u_x @ s_x @ z_x_estimated + u_e @ s_e @ z_e_estimated).squeeze(-1) + mu_x[None] - x)
            # print('###check optimality###')
            # print(f'grad_x: {grad_x}')
            # print(f'grad_e: {grad_e}')
            # print(f'recon error: {recon_err}')
        return {'x': kp_reg_x, 'e': kp_reg_e, 'k_e': k_e}

    def print_statistics(self):
        for i, sec in enumerate(self.sections):
            print(f"mu_x_{i}: {self.getattr(f'mu_x_{i}')}")
            print(f"u_x_{i}: {self.getattr(f'u_x_{i}')}")
            print(f"s_x_{i}: {self.getattr(f's_x_{i}')}")
            print(f"u_e_{i}: {self.getattr(f'u_e_{i}')}")
            print(f"s_e_{i}: {self.getattr(f's_e_{i}')}")
            
    def export_statistics(self):
        d = {}
        d['sections'] = self.sections.copy()
        for i, sec in enumerate(self.sections):
            d[f"mu_x_{i}"] = self.getattr(f'mu_x_{i}')
            d[f"u_x_{i}"] = self.getattr(f'u_x_{i}')
            d[f"s_x_{i}"] = self.getattr(f's_x_{i}')
            d[f"u_e_{i}"] = self.getattr(f'u_e_{i}')
            d[f"s_e_{i}"] = self.getattr(f's_e_{i}')
        return d
    
    def extract_emotion(self, kp):
        kp_reg_xs = []
        kp_reg_es = []
        k_es = []
        
        # secs, means = self.split_section_and_normalize(kp['value'])
        # means = [torch.zeros_like(m) for m in means]
        secs = self.split_section(kp['value'])
        
        for i, sec in enumerate(secs):
            mu_x = self.getattr(f'mu_x_{i}')
            u_x = self.getattr(f'u_x_{i}')
            s_x = self.getattr(f's_x_{i}')
            u_e = self.getattr(f'u_e_{i}')
            s_e = self.getattr(f's_e_{i}')
            sigma_err = self.getattr(f'sigma_err_{i}')
            
            kp_reg = self.extract_emotion_section(sec.flatten(1), self.scalers[i], mu_x, u_x, s_x, u_e, s_e, sigma_err)
            kp_reg_xs.append(kp_reg['x'])
            kp_reg_es.append(kp_reg['e'])
            k_es.append(kp_reg['k_e'])
        
        return {'x': kp_reg_xs, 'e': kp_reg_es, 'k_e': k_es}        
        # return {'x': kp_reg_xs, 'e': kp_reg_es, 'k_e': k_es, 'mean': means}
        
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
        # generated['source_x'] = self.concat_section_and_denormalize([v.view(bs, -1, 3) for v in source_res['x']], source_res['mean'])
        # generated['driving_x'] = self.concat_section_and_denormalize([v.view(bs, -1, 3) for v in driving_res['x']], driving_res['mean'])
        generated['source_x'] = self.concat_section([v.view(bs, -1, 3) for v in source_res['x']])
        generated['driving_x'] = self.concat_section([v.view(bs, -1, 3) for v in driving_res['x']])
        generated['source_e'] = self.concat_section(source_res['e']).view(bs, -1 ,3)
        generated['driving_e'] = self.concat_section(driving_res['e']).view(bs, -1, 3)
        # generated['mesh_bias'] = self.get_mesh_bias()[None].repeat(bs, 1, 1)
        generated['source_scale'] = self.concat_section(source_res['k_e'])
        generated['driving_scale'] = self.concat_section(driving_res['k_e'])
        
        return loss_values, generated
        
        
    def estimate_params_section_v3(self, x, R_src, R, e_normalized, scaler, mu_x, u_x, s_x, u_e, s_e, sigma_err, num_kp, threshold=None, var_coef=1):
        # x: B x N * 3   
        
        assert x is not None
        
        # x: B x N * 3   
        # R: B x 3 x 3
        
        ### Src ###
        assert x is not None
        
        print(f'R_src shape: {R_src.shape}')
        ignore_axis = R_src[:, :, [2]].cuda() # B x 3 x 1
        ignore_matrix = ignore_axis @ ignore_axis.transpose(1, 2) # B x 3 x 3
        filter_matrix = torch.eye(3)[None].cuda() - ignore_matrix # B x 3 x 3
        P = ignore_matrix
        F = filter_matrix
        N = x.shape[1] // 3
        B = x.shape[0]
        # print(f'filter matrix shape: {filter_matrix.shape}')
        # print(f'N: {N}')
        x = x - mu_x[None]
        k_e = self.get_scale(x, scaler).mean(dim=0) # n * 3
        x = x.unsqueeze(2)
        
        u_e = k_e.unsqueeze(1) * u_e    # 3 * num_kp x num_pc
        num_items = len(x)
        num_pc = u_e.size(1)
        
        A_e = u_e @ s_e
        A_x = u_x @ s_x
        sigma_err_i = (P.transpose(1, 2).matmul(P) / (1 * self.s_err_square) + F.transpose(1, 2).matmul(F) / (1 * self.s_err_square)).inverse() # B x 3 x 3
        # print(f'sigma_err_i shape: {sigma_err_i.shape}')
        # print(f' shape: {sigma_err_i.shape}')
        sigma_err_i = block_diagonal_batch(sigma_err_i, N)
        sigma_e_i = (A_e.t().matmul(sigma_err_i).matmul(A_e) + torch.eye(A_e.size(1)).cuda()[None]).inverse()
        A_tilda_x_i = -sigma_e_i.matmul(A_e.t()).matmul(sigma_err_i.inverse()).matmul(A_x)
        B_tilda_x_i = -sigma_e_i.matmul(A_e.t()).matmul(sigma_err_i.inverse()).matmul(x)
        A_star_x_i = A_e.matmul(A_tilda_x_i) + A_x[None]
        B_star_x_i = A_e.matmul(B_tilda_x_i) + x
        # print(f'A_tilda_x_i: {A_tilda_x_i.shape}')
        # print(f'B_tilda_x_i: {B_tilda_x_i.shape}')
        # print(f'A_star_x_i: {A_star_x_i.shape}')
        # print(f'B_star_x_i: {B_star_x_i.shape}')
        # print(f'sigma_err_i: {sigma_err_i.shape}')
        # print(f'sigma_e_i: {sigma_e_i.shape}')
        sigma_x_i = (A_star_x_i.transpose(1, 2).matmul(sigma_err_i.inverse()).matmul(A_star_x_i) + A_tilda_x_i.transpose(1, 2) @ A_tilda_x_i).inverse()
        mu_x_i = sigma_x_i.matmul(A_star_x_i.transpose(1, 2).matmul(sigma_err_i.inverse()).matmul(B_star_x_i) + A_tilda_x_i.transpose(1, 2).matmul(B_tilda_x_i))
        sigma_x_total = sigma_x_i.inverse().sum(dim=0).inverse()
        mu_x_total = sigma_x_total.matmul(sigma_x_i.inverse().matmul(mu_x_i).sum(dim=0))
        
        sigma_x_final = (sigma_x_total.inverse() / num_items + var_coef * torch.eye(num_pc).cuda()).inverse()
        mu_x_final = sigma_x_final.matmul(sigma_x_total.inverse().matmul(mu_x_total) / num_items)
        sigma_x_src = sigma_x_final 
        mu_x_src = mu_x_final
        # print(f'posterior: {sigma_x_src}, {mu_x_src}')
        
        # sigma_x_src = torch.eye(len(sigma_x_src)).cuda()
        # mu_x_src = torch.zeros_like(mu_x_src)
        # print(f'prior: {sigma_x_src}, {mu_x_src}')
        
        # print(f'R shape: {R.shape}')
        ignore_axis = R[:, :, [2]].cuda() # B x 3 x 1
        ignore_matrix = ignore_axis @ ignore_axis.transpose(1, 2) # B x 3 x 3
        filter_matrix = torch.eye(3)[None].cuda() - ignore_matrix # B x 3 x 3
        P = ignore_matrix
        F = filter_matrix
        N = x.shape[1] // 3
        B = x.shape[0]
        # print(f'filter matrix shape: {filter_matrix.shape}')
        # print(f'N: {N}')
        
        num_items = len(R)
        num_pc = u_e.size(1)
        
        e_driven = e_normalized * k_e[None]
        # print(f'e_normalized shape: {e_normalized.shape}')
        A_e = u_e @ s_e
        A_x = u_x @ s_x
        sigma_err_i = (P.transpose(1, 2).matmul(P) / (self.s_err_square) + F.transpose(1, 2).matmul(F) / (self.s_err_square)).inverse() # B x 3 x 3
        # print(f'sigma_err_i shape: {sigma_err_i.shape}')
        # print(f' shape: {sigma_err_i.shape}')
        sigma_err_i = block_diagonal_batch(sigma_err_i, N)
        sigma_e_i = (A_e.t().matmul(sigma_err_i).matmul(A_e) + torch.eye(A_e.size(1)).cuda()[None]).inverse()
        A_tilda_x_i = -sigma_e_i.matmul(A_e.t()).matmul(sigma_err_i.inverse()).matmul(A_x)
        B_tilda_x_i = -sigma_e_i.matmul(A_e.t()).matmul(sigma_err_i.inverse()).matmul(x)
        A_star_x_i = A_e.matmul(A_tilda_x_i) + A_x[None]
        B_star_x_i = A_e.matmul(B_tilda_x_i) + x
        # print(f'A_tilda_x_i: {A_tilda_x_i.shape}')
        # print(f'B_tilda_x_i: {B_tilda_x_i.shape}')
        # print(f'A_star_x_i: {A_star_x_i.shape}')
        # print(f'B_star_x_i: {B_star_x_i.shape}')
        # print(f'sigma_err_i: {sigma_err_i.shape}')
        # print(f'sigma_e_i: {sigma_e_i.shape}')
        sigma_x_i = (A_star_x_i.transpose(1, 2).matmul(sigma_err_i.inverse()).matmul(A_star_x_i) + A_tilda_x_i.transpose(1, 2) @ A_tilda_x_i).inverse()
        mu_x_i = sigma_x_i.matmul(A_star_x_i.transpose(1, 2).matmul(sigma_err_i.inverse()).matmul(B_star_x_i) + A_tilda_x_i.transpose(1, 2).matmul(B_tilda_x_i))
        sigma_x_total = sigma_x_i.inverse().sum(dim=0).inverse()
        mu_x_total = sigma_x_total.matmul(sigma_x_i.inverse().matmul(mu_x_i).sum(dim=0))
        sigma_x_final = (sigma_x_total.inverse() / num_items + sigma_x_src.inverse()).inverse()
        print(f'pre sigma: {num_items * sigma_x_total}')
        print(f'post sigma: {sigma_x_final}')
        mu_x_final = sigma_x_final.matmul(sigma_x_total.inverse().matmul(mu_x_total) / num_items)
        A_hat_i = - sigma_e_i.matmul(A_e.t()).matmul(sigma_err_i.inverse())
        A_ohm_i = A_star_x_i.transpose(1, 2).matmul(sigma_err_i.inverse()).matmul(A_e.matmul(A_hat_i) + torch.eye(len(A_e)).cuda()[None].repeat(len(A_hat_i), 1, 1)) + A_tilda_x_i.transpose(1, 2).matmul(A_hat_i)
        print(f'A_ha_i shape: {A_hat_i.shape}')
        print(f'A_ohm_i shape: {A_ohm_i.shape}')
        w = (torch.eye(len(sigma_x_final)).cuda() - 1 / num_items * sigma_x_final @ (A_ohm_i.matmul(A_x).sum(0))).inverse().matmul(sigma_x_final).matmul(torch.einsum('bkn,bn->bk',A_ohm_i, e_driven).sum(0) / num_items + sigma_x_src.inverse().matmul(mu_x_src).squeeze(1))
        print(f'w shape: {w.shape}')
        # sigma_inverse = filter_matrix.transpose(1, 2) @ (torch.eye(A_e.size(0)).cuda() - A_e @ sigma_e @ A_e.t()) @ filter_matrix
        # sigma_x = (A_x.t() @ sigma_inverse @ A_x).inverse()
        # mu_xs = sigma_x @ A_x.t() @ sigma_inverse @ x.unsqueeze(2) # B x num_pc x 1
        # sigma_x_total = (sigma_x.inverse() / self.s_err_square + torch.eye(sigma_x.size(0)).cuda()).inverse()
        # # sigma_x_total = (len(x) * sigma_x.inverse() / self.s_err_square + torch.eye(sigma_x.size(0)).cuda()).inverse()
        # mu_x_total = sigma_x_total @ sigma_x.inverse() @ mu_xs.sum(dim=0) / (self.s_err_square * len(x)) # 3 * num_kp x 1
        
        
        
        # sigma_inverse = torch.eye(A_e.size(0))
        # sigma_x = ()
        # sigma_e_square_inverse = u_e @ (u_e.t() @ u_e).inverse() @ s_e.inverse().t() @ s_e.inverse() @ (u_e.t() @ u_e).inverse() @ u_e.t()
        # sigma_x_square_inverse = u_x @ s_x.inverse().t() @ s_x.inverse() @ u_x.t()
        # sigma_inverse = (sigma_err.inverse() + sigma_e_square_inverse).inverse()
        # sigma_inverse = sigma_err.inverse() - sigma_err.inverse() @ sigma_inverse @ sigma_err.inverse()
        # sigma_x_X_square_inverse = sigma_x_square_inverse + \
        #     len(x) * sigma_inverse
        # mu_x_X = sigma_x_X_square_inverse.inverse() @ (0 * sigma_x_square_inverse @ mu_x.unsqueeze(1) + \
        #     torch.stack([((torch.eye(len(sigma_err)).cuda() - (torch.eye(len(sigma_err)).cuda() + sigma_e_square_inverse @ sigma_err).inverse()) @ sigma_err.inverse() @ X.unsqueeze(1)) for X in x], dim=0).sum(dim=0))
        
        # print(f'test: {sigma_x_X_square_inverse.inverse() @ sigma_x_X_square_inverse}')
        # mu_x_X = mu_x.unsqueeze(1)
        # mu_x_X = x[0].unsqueeze(1)
        # print(f'k_e: {mu_x_X}')
        # print(f'simga x square inverse: {sigma_e_square_inverse}')
        # print(f'simga e square inverse: {sigma_e_square_inverse}')
        # print(f'simga e square inverse: {sigma_e_square_inverse}')
        # print(f'simga e square inverse: {sigma_e_square_inverse}')
        kp_reg_e = x.squeeze(2) - (A_x.matmul(w))[None]   # B x N * 3
        kp_reg_x = (A_x.matmul(w)) + mu_x   # N * 3

        return {'x': kp_reg_x, 'e': kp_reg_e, 'k_e': k_e}
   
      
    def estimate_params_v3(self, x, drv_data, e_normalized, threshold=None):
        # data: {v: B x ...}
        bs = len(x['mesh']['value'])
    
        kp = x['mesh']
        
        kp_reg_xs = []
        kp_reg_es = []
        k_es = []
        
        secs = self.split_section(kp['value'].cuda())
        
        for i, sec in enumerate(secs):
            mu_x = self.getattr(f'mu_x_{i}')
            u_x = self.getattr(f'u_x_{i}')
            s_x = self.getattr(f's_x_{i}')
            u_e = self.getattr(f'u_e_{i}')
            s_e = self.getattr(f's_e_{i}')
            sigma_err = self.getattr(f'sigma_err_{i}')
            
            kp_reg = self.estimate_params_section_v3(sec.flatten(1), kp['R'], drv_data['mesh']['R'], e_normalized[i], self.scalers[i], mu_x, u_x, s_x, u_e, s_e, sigma_err, self.sections[i][1], threshold=threshold, var_coef=1 if i + 1 != len(secs) else 10000)
            
            kp_reg_xs.append(kp_reg['x'])   # num_kp * 3
            kp_reg_es.append(kp_reg['e'])   # B x num_kp * 3
            k_es.append(kp_reg['k_e'])      # num_kp * 3
        
        return {'x': kp_reg_xs, 'e': kp_reg_es, 'k_e': k_es}
    
    def estimate_params_section_v2(self, x, R, e_normalized, scaler, mu_x, u_x, s_x, u_e, s_e, sigma_err, num_kp, threshold=None):
        # x: B x N * 3   
        
        assert x is not None
        
        print(f'R shape: {R.shape}')
        ignore_axis = R[:, :, [2]] # B x 3 x 1
        ignore_matrix = ignore_axis @ ignore_axis.transpose(1, 2) # B x 3 x 3
        filter_matrix = torch.eye(3)[None].cuda() - ignore_matrix # B x 3 x 3
        N = x.shape[1] // 3
        B = x.shape[0]
        print(f'filter matrix shape: {filter_matrix.shape}')
        
        x = x - mu_x[None]
        k_e = self.get_scale(x, scaler).mean(dim=0) # n * 3
        
        u_e = k_e.unsqueeze(1) * u_e    # 3 * num_kp x num_pc
        
        A_e = u_e @ s_e
        A_x = u_x @ s_x
        sigma_e = (A_e.t() @ A_e + torch.eye(A_e.size(1)).cuda() * self.s_err_square).inverse()
        sigma_inverse = torch.eye(A_e.size(0)).cuda() - A_e @ sigma_e @ A_e.t()

        ################
        sigma_x = (A_x.t() @ sigma_inverse @ A_x).inverse()
        mu_xs = sigma_x @ A_x.t() @ sigma_inverse @ x.unsqueeze(2) # B x num_pc x 1
        sigma_x_total = (len(x) * sigma_x.inverse() + torch.eye(sigma_x.size(0)).cuda() * self.s_err_square).inverse()
        mu_x_total = sigma_x_total @ sigma_x.inverse() @ mu_xs.sum(dim=0) # 3 * num_kp x 1
        sigma_x_total = sigma_x_total * self.s_err_square
    
        # mu_x_total = torch.zeros(num_kp, 1).cuda()
        # sigma_x_total = torch.eye(num_kp).cuda()
        
        # v2
        e_driven = e_normalized * k_e[None]
        n = len(e_driven)
        # sigma_x_driven = (n * sigma_x.inverse() / self.s_err_square + sigma_x_total.inverse()).inverse()
        sigma_x_driven = (sigma_x.inverse() / self.s_err_square + sigma_x_total.inverse()).inverse()
        M = sigma_x @ A_x.t() @ sigma_inverse
        N = mu_x_total.t() @ sigma_x_total.inverse()
        zx_driven = (torch.eye(num_kp).cuda() - sigma_x_driven.t() @ sigma_x.inverse().t() @ M @ A_x / self.s_err_square).inverse() @ sigma_x_driven.t() @ (sigma_x.inverse().t() @ M @ e_driven.mean(dim=0).unsqueeze(1) / self.s_err_square + N.t()) 
        x_driven = A_x @ zx_driven
        # print(f'M shape: {M.shape}')
        # print(f'N shape: {N.shape}')
        # print(f'x_driven shape: {x_driven.shape}')
        # sigma_inverse = torch.eye(A_e.size(0))
        # sigma_x = ()
        # sigma_e_square_inverse = u_e @ (u_e.t() @ u_e).inverse() @ s_e.inverse().t() @ s_e.inverse() @ (u_e.t() @ u_e).inverse() @ u_e.t()
        # sigma_x_square_inverse = u_x @ s_x.inverse().t() @ s_x.inverse() @ u_x.t()
        # sigma_inverse = (sigma_err.inverse() + sigma_e_square_inverse).inverse()
        # sigma_inverse = sigma_err.inverse() - sigma_err.inverse() @ sigma_inverse @ sigma_err.inverse()
        # sigma_x_X_square_inverse = sigma_x_square_inverse + \
        #     len(x) * sigma_inverse
        # mu_x_X = sigma_x_X_square_inverse.inverse() @ (0 * sigma_x_square_inverse @ mu_x.unsqueeze(1) + \
        #     torch.stack([((torch.eye(len(sigma_err)).cuda() - (torch.eye(len(sigma_err)).cuda() + sigma_e_square_inverse @ sigma_err).inverse()) @ sigma_err.inverse() @ X.unsqueeze(1)) for X in x], dim=0).sum(dim=0))
        
        # print(f'test: {sigma_x_X_square_inverse.inverse() @ sigma_x_X_square_inverse}')
        # mu_x_X = mu_x.unsqueeze(1)
        # mu_x_X = x[0].unsqueeze(1)
        # print(f'k_e: {mu_x_X}')
        # print(f'simga x square inverse: {sigma_e_square_inverse}')
        # print(f'simga e square inverse: {sigma_e_square_inverse}')
        # print(f'simga e square inverse: {sigma_e_square_inverse}')
        # print(f'simga e square inverse: {sigma_e_square_inverse}')
        

        kp_reg_e = x - x_driven.squeeze(1)[None]   # B x N * 3
        
        # thresholding
        if threshold is not None:
            reg_ze = (self.s_err_square * torch.eye(A_e.size(1)).cuda() + A_e.t() @ A_e).inverse() @ A_e.t() @ kp_reg_e.squeeze(0).unsqueeze(1) # num_kp x 1
            chi2_value = (reg_ze ** 2).sum()
            chi2_threshold = stats.chi2(df=len(reg_ze)).ppf(threshold)
            print(f'Source Mesh Score: {stats.chi2(df=len(reg_ze)).cdf(chi2_value.item())}')
            if chi2_value <= chi2_threshold:
                print('Source Mesh is treated as Neutralized Mesh')
                x_driven = x.squeeze(0).unsqueeze(1)
                kp_reg_e = x - x_driven.squeeze(1)[None]   # B x N * 3
                
        kp_reg_x = x_driven.squeeze(1) + mu_x   # N * 3

        return {'x': kp_reg_x, 'e': kp_reg_e, 'k_e': k_e}
      
    def estimate_params_v2(self, x, e_normalized, threshold=None):
        # data: {v: B x ...}
        bs = len(x['mesh']['value'])
    
        kp = x['mesh']
        
        kp_reg_xs = []
        kp_reg_es = []
        k_es = []
        
        secs = self.split_section(kp['value'].cuda())
        
        for i, sec in enumerate(secs):
            mu_x = self.getattr(f'mu_x_{i}')
            u_x = self.getattr(f'u_x_{i}')
            s_x = self.getattr(f's_x_{i}')
            u_e = self.getattr(f'u_e_{i}')
            s_e = self.getattr(f's_e_{i}')
            sigma_err = self.getattr(f'sigma_err_{i}')
            
            kp_reg = self.estimate_params_section_v2(sec.flatten(1), kp['R'], e_normalized[i], self.scalers[i], mu_x, u_x, s_x, u_e, s_e, sigma_err, self.sections[i][1], threshold=threshold)
            
            kp_reg_xs.append(kp_reg['x'])   # num_kp * 3
            kp_reg_es.append(kp_reg['e'])   # B x num_kp * 3
            k_es.append(kp_reg['k_e'])      # num_kp * 3
        
        return {'x': kp_reg_xs, 'e': kp_reg_es, 'k_e': k_es}
    
    
    def estimate_params_section(self, x, R, scaler, mu_x, u_x, s_x, u_e, s_e, sigma_err, num_pc, var_coef=1):
        # x: B x N * 3   
        # R: B x 3 x 3
        
        assert x is not None
        
        print(f'R shape: {R.shape}')
        ignore_axis = R[:, :, [2]].cuda() # B x 3 x 1
        ignore_matrix = ignore_axis @ ignore_axis.transpose(1, 2) # B x 3 x 3
        filter_matrix = torch.eye(3)[None].cuda() - ignore_matrix # B x 3 x 3
        P = ignore_matrix
        F = filter_matrix
        N = x.shape[1] // 3
        B = x.shape[0]
        # print(f'filter matrix shape: {filter_matrix.shape}')
        # print(f'N: {N}')
        x = x - mu_x[None]
        k_e = self.get_scale(x, scaler).mean(dim=0) # n * 3
        x = x.unsqueeze(2)
        
        u_e = k_e.unsqueeze(1) * u_e    # 3 * num_kp x num_pc
        num_items = len(x)
        num_pc = u_e.size(1)
        
        A_e = u_e @ s_e
        A_x = u_x @ s_x
        sigma_err_i = (P.transpose(1, 2).matmul(P) / (1 * self.s_err_square) + F.transpose(1, 2).matmul(F) / self.s_err_square).inverse() # B x 3 x 3
        # print(f'sigma_err_i shape: {sigma_err_i.shape}')
        # print(f' shape: {sigma_err_i.shape}')
        sigma_err_i = block_diagonal_batch(sigma_err_i, N)
        sigma_e_i = (A_e.t().matmul(sigma_err_i).matmul(A_e) + torch.eye(A_e.size(1)).cuda()[None]).inverse()
        A_tilda_x_i = -sigma_e_i.matmul(A_e.t()).matmul(sigma_err_i.inverse()).matmul(A_x)
        B_tilda_x_i = -sigma_e_i.matmul(A_e.t()).matmul(sigma_err_i.inverse()).matmul(x)
        A_star_x_i = A_e.matmul(A_tilda_x_i) + A_x[None]
        B_star_x_i = A_e.matmul(B_tilda_x_i) + x
        # print(f'A_tilda_x_i: {A_tilda_x_i.shape}')
        # print(f'B_tilda_x_i: {B_tilda_x_i.shape}')
        # print(f'A_star_x_i: {A_star_x_i.shape}')
        # print(f'B_star_x_i: {B_star_x_i.shape}')
        # print(f'sigma_err_i: {sigma_err_i.shape}')
        # print(f'sigma_e_i: {sigma_e_i.shape}')
        sigma_x_i = (A_star_x_i.transpose(1, 2).matmul(sigma_err_i.inverse()).matmul(A_star_x_i) + A_tilda_x_i.transpose(1, 2) @ A_tilda_x_i).inverse()
        mu_x_i = sigma_x_i.matmul(A_star_x_i.transpose(1, 2).matmul(sigma_err_i.inverse()).matmul(B_star_x_i) + A_tilda_x_i.transpose(1, 2).matmul(B_tilda_x_i))
        sigma_x_total = sigma_x_i.inverse().sum(dim=0).inverse()
        mu_x_total = sigma_x_total.matmul(sigma_x_i.inverse().matmul(mu_x_i).sum(dim=0))
        sigma_x_final = (sigma_x_total.inverse() / num_items + var_coef * torch.eye(num_pc).cuda()).inverse()
        mu_x_final = sigma_x_final.matmul(sigma_x_total.inverse().matmul(mu_x_total) / num_items)
        
        # sigma_inverse = filter_matrix.transpose(1, 2) @ (torch.eye(A_e.size(0)).cuda() - A_e @ sigma_e @ A_e.t()) @ filter_matrix
        # sigma_x = (A_x.t() @ sigma_inverse @ A_x).inverse()
        # mu_xs = sigma_x @ A_x.t() @ sigma_inverse @ x.unsqueeze(2) # B x num_pc x 1
        # sigma_x_total = (sigma_x.inverse() / self.s_err_square + torch.eye(sigma_x.size(0)).cuda()).inverse()
        # # sigma_x_total = (len(x) * sigma_x.inverse() / self.s_err_square + torch.eye(sigma_x.size(0)).cuda()).inverse()
        # mu_x_total = sigma_x_total @ sigma_x.inverse() @ mu_xs.sum(dim=0) / (self.s_err_square * len(x)) # 3 * num_kp x 1
        
        
        
        # sigma_inverse = torch.eye(A_e.size(0))
        # sigma_x = ()
        # sigma_e_square_inverse = u_e @ (u_e.t() @ u_e).inverse() @ s_e.inverse().t() @ s_e.inverse() @ (u_e.t() @ u_e).inverse() @ u_e.t()
        # sigma_x_square_inverse = u_x @ s_x.inverse().t() @ s_x.inverse() @ u_x.t()
        # sigma_inverse = (sigma_err.inverse() + sigma_e_square_inverse).inverse()
        # sigma_inverse = sigma_err.inverse() - sigma_err.inverse() @ sigma_inverse @ sigma_err.inverse()
        # sigma_x_X_square_inverse = sigma_x_square_inverse + \
        #     len(x) * sigma_inverse
        # mu_x_X = sigma_x_X_square_inverse.inverse() @ (0 * sigma_x_square_inverse @ mu_x.unsqueeze(1) + \
        #     torch.stack([((torch.eye(len(sigma_err)).cuda() - (torch.eye(len(sigma_err)).cuda() + sigma_e_square_inverse @ sigma_err).inverse()) @ sigma_err.inverse() @ X.unsqueeze(1)) for X in x], dim=0).sum(dim=0))
        
        # print(f'test: {sigma_x_X_square_inverse.inverse() @ sigma_x_X_square_inverse}')
        # mu_x_X = mu_x.unsqueeze(1)
        # mu_x_X = x[0].unsqueeze(1)
        # print(f'k_e: {mu_x_X}')
        # print(f'simga x square inverse: {sigma_e_square_inverse}')
        # print(f'simga e square inverse: {sigma_e_square_inverse}')
        # print(f'simga e square inverse: {sigma_e_square_inverse}')
        # print(f'simga e square inverse: {sigma_e_square_inverse}')
        # print(f'mu_x_final shape: {mu_x_final.shape}')
        # print(f'mu_x_final shape: {mu_x_final.shape}')
        kp_reg_e = x.squeeze(2) - (A_x @ mu_x_final).squeeze(1)[None]   # B x N * 3
        kp_reg_x = (A_x @ mu_x_final.squeeze(1)) + mu_x   # N * 3

        return {'x': kp_reg_x, 'e': kp_reg_e, 'k_e': k_e}
      
    def estimate_params(self, x):
        # data: {v: B x ...}
        bs = len(x['mesh']['value'])
    
        kp = x['mesh']
        
        kp_reg_xs = []
        kp_reg_es = []
        k_es = []
        
        secs = self.split_section(kp['value'].cuda())
        
        for i, sec in enumerate(secs):
            mu_x = self.getattr(f'mu_x_{i}')
            u_x = self.getattr(f'u_x_{i}')
            s_x = self.getattr(f's_x_{i}')
            u_e = self.getattr(f'u_e_{i}')
            s_e = self.getattr(f's_e_{i}')
            sigma_err = self.getattr(f'sigma_err_{i}')
            
            kp_reg = self.estimate_params_section(sec.flatten(1), kp['R'], self.scalers[i], mu_x, u_x, s_x, u_e, s_e, sigma_err, self.sections[0][1], var_coef=10000 if i + 1 != len(secs) else 10000)
            
            kp_reg_xs.append(kp_reg['x'])   # num_kp * 3
            kp_reg_es.append(kp_reg['e'])   # B x num_kp * 3
            k_es.append(kp_reg['k_e'])      # num_kp * 3
        
        return {'x': kp_reg_xs, 'e': kp_reg_es, 'k_e': k_es}
    
    def drive(self, drv, x_src, k_e_src, x_drv=None, k_e_drv=None):
        # drv: {v: 1 x ...}
        secs = self.split_section(drv['mesh']['value'].cuda())
        src = []
        driving = []
        driven = []
        for i, sec in enumerate(secs):
            # sec: 1 x N * 3
            assert x_drv is not None
            assert k_e_drv is not None
            sec = sec[0]    # N x 3
            u_e = self.getattr(f'u_e_{i}')
            s_e = self.getattr(f's_e_{i}')
            u_e_drv = k_e_drv[i].unsqueeze(1) * u_e
            A_e_drv = u_e_drv @ s_e
            e_drv = sec.flatten(0) - x_drv[i]
            coef = A_e_drv.t()
            ### ignore z ###
            # if len(driven) >=2:
            #     momentum = driven[-1] - driven[-2] # N x 3
                
            R = drv['mesh']['R'][0].cuda() # 3 x 3
            print(f'R shape: {R.shape}')
            ignore_axis = R[:, [2]] # 3 x 1
            ignore_matrix = ignore_axis @ ignore_axis.t() # 3 x 3
            filter_matrix = torch.eye(3).cuda() - ignore_matrix # 3 x 3
            P = ignore_matrix
            F = filter_matrix
            N = sec.shape[0]
            num_kp = A_e_drv.size(1)
            print(f'filter matrix shape: {filter_matrix.shape}')
            sigma_err_i = (P.t() @ P / (100 * self.s_err_square) + F.t() @ F / self.s_err_square).inverse()
            sigma_err_i = block_diagonal_batch(sigma_err_i[None], N).squeeze(0)
            sigma_e_i = (A_e_drv.t() @ sigma_err_i.inverse() @ A_e_drv + torch.eye(num_kp).cuda()).inverse()
            mu_e_i = sigma_e_i @ A_e_drv.t() @ sigma_err_i.inverse() @ e_drv.unsqueeze(1)
            z_e = mu_e_i
            
            
            # z_e = (u_e_drv.t() @ u_e_drv).inverse() @ u_e_drv.t() @ e_drv.unsqueeze(1) # N * 3 x 1
            
            u_e_src = k_e_src[i].unsqueeze(1) * u_e
            # e_src = e_drv * k_e_src[i] / k_e_drv[i] 
            e_src = u_e_src @ s_e @ z_e
            X_driven = x_src[i].view(-1, 3) + e_src.view(-1, 3)
            # print(f'e src shape: {e_src.shape}')
            # print(f'mean src shape: {mean_src[i].shape}')
            # print(f'sec shape: {sec.shape}')
            # print(f'X_driven shape: {X_driven.shape}')
            # print(f'driving shape: {(x_drv[i].view(-1, 3) + means[i][None]).shape}')
            # print(f'source shape: {(x_src[i].view(-1, 3) + mean_src[i][None]).shape}')
            driven.append(X_driven)
            driving.append(x_drv[i].view(-1, 3) + e_drv.view(-1, 3))
            src.append(x_src[i].view(-1, 3))
            
        
        return torch.cat(src, dim=0), torch.cat(driving, dim=0), torch.cat(driven, dim=0)
            
   