import torch
import torch.nn as nn
import torch.nn.functional as functional
import numpy as np
import torch.nn.functional as F
import os
import math
import pointnet2_ops_lib.pointnet2_ops.pointnet2_utils as pointnet2_utils
from model.utils import square_distance, index_points, farthest_point_sample


class CrossTransformerBlock(nn.Module):
    def __init__(self, dim_inp, dim, nneigh=7, reduce_dim=True, separate_delta=True):
        super().__init__()

        # dim_inp = dim
        # dim = dim  # // 2
        self.dim = dim

        self.nneigh = nneigh
        self.separate_delta = separate_delta

        self.fc_delta = nn.Sequential(
            nn.Linear(3, dim),
            nn.ReLU(),
            nn.Linear(dim, dim)
        )
        self.fc_gamma = nn.Sequential(
            nn.Linear(dim, dim),
            nn.ReLU(),
            nn.Linear(dim, dim)
        )
        self.w_k_global = nn.Linear(dim_inp, dim, bias=False)
        self.w_v_global = nn.Linear(dim_inp, dim, bias=False)

        self.w_qs = nn.Linear(dim_inp, dim, bias=False)
        self.w_ks = nn.Linear(dim_inp, dim, bias=False)
        self.w_vs = nn.Linear(dim_inp, dim, bias=False)

        if not reduce_dim:
            self.fc = nn.Linear(dim, dim_inp)
        self.reduce_dim = reduce_dim

    # xyz_q: B x n_queries x 3
    # lat_rep: B x dim
    # xyz: B x n_anchors x 3,
    # points: B x n_anchors x dim
    def forward(self, xyz_q, lat_rep, xyz, points):
        with torch.no_grad():
            dists = square_distance(xyz_q, xyz)
            ## knn group
            knn_idx = dists.argsort()[:, :, :self.nneigh]  # b x nQ x k
            #print(knn_idx.shape)

            #knn = KNN(k=self.nneigh, transpose_mode=True)
            #_, knn_idx = knn(xyz, xyz_q)  # B x npoint x K
            ##
            #print(knn_idx.shape)

        b, nQ, _ = xyz_q.shape
        # b, nK, dim = points.shape

        if len(lat_rep.shape) == 2:
            q_attn = self.w_qs(lat_rep).unsqueeze(1).repeat(1, nQ, 1)
            k_global = self.w_k_global(lat_rep).unsqueeze(1).repeat(1, nQ, 1).unsqueeze(2)
            v_global = self.w_v_global(lat_rep).unsqueeze(1).repeat(1, nQ, 1).unsqueeze(2)
        else:
            q_attn = self.w_qs(lat_rep)
            k_global = self.w_k_global(lat_rep).unsqueeze(2)
            v_global = self.w_v_global(lat_rep).unsqueeze(2)

        k_attn = index_points(self.w_ks(points),
                              knn_idx)  # b, nQ, k, dim  # self.w_ks(points).unsqueeze(1).repeat(1, nQ, 1, 1)
        k_attn = torch.cat([k_attn, k_global], dim=2)
        v_attn = index_points(self.w_vs(points), knn_idx)  # #self.w_vs(points).unsqueeze(1).repeat(1, nQ, 1, 1)
        v_attn = torch.cat([v_attn, v_global], dim=2)
        xyz = index_points(xyz, knn_idx)  # xyz = xyz.unsqueeze(1).repeat(1, nQ, 1, 1)
        pos_encode = self.fc_delta(xyz_q[:, :, None] - xyz)  # b x nQ x k x dim
        pos_encode = torch.cat([pos_encode, torch.zeros([b, nQ, 1, self.dim], device=pos_encode.device)],
                               dim=2)  # b, nQ, k+1, dim
        if self.separate_delta:
            pos_encode2 = self.fc_delta(xyz_q[:, :, None] - xyz)  # b x nQ x k x dim
            pos_encode2 = torch.cat([pos_encode2, torch.zeros([b, nQ, 1, self.dim], device=pos_encode2.device)],
                                   dim=2)  # b, nQ, k+1, dim
        else:
            pos_encode2 = pos_encode

        attn = self.fc_gamma(q_attn[:, :, None] - k_attn + pos_encode)
        attn = functional.softmax(attn, dim=-2)  # b x nQ x k+1 x dim

        res = torch.einsum('bmnf,bmnf->bmf', attn, v_attn + pos_encode2)  # b x nQ x dim

        if not self.reduce_dim:
            res = self.fc(res)
        return res



class ResnetBlockFC(nn.Module):
    ''' Fully connected ResNet Block class.
    Copied from https://github.com/autonomousvision/convolutional_occupancy_networks

    Args:
        size_in (int): input dimension
        size_out (int): output dimension
        size_h (int): hidden dimension
    '''

    def __init__(self, size_in, size_out=None, size_h=None):
        super().__init__()
        # Attributes
        if size_out is None:
            size_out = size_in

        if size_h is None:
            size_h = min(size_in, size_out)

        self.size_in = size_in
        self.size_h = size_h
        self.size_out = size_out
        # Submodules
        self.fc_0 = nn.Linear(size_in, size_h)
        self.fc_1 = nn.Linear(size_h, size_out)
        self.actvn = nn.ReLU()

        if size_in == size_out:
            self.shortcut = None
        else:
            self.shortcut = nn.Linear(size_in, size_out, bias=False)
        # Initialization
        nn.init.zeros_(self.fc_1.weight)

    def forward(self, x):
        net = self.fc_0(self.actvn(x))
        dx = self.fc_1(self.actvn(net))

        if self.shortcut is not None:
            x_s = self.shortcut(x)
        else:
            x_s = x

        return x_s + dx


