'''
AIR-Nets
Author: Simon Giebenhain
Code: https://github.com/SimonGiebenhain/AIR-Nets
'''

import torch
import torch.nn as nn
import torch.nn.functional as functional
import numpy as np
from time import time
import torch.nn.functional as F
import os
import math
import pointnet2_ops_lib.pointnet2_ops.pointnet2_utils as pointnet2_utils
from model.utils import square_distance, index_points, farthest_point_sample

class TransitionDown(nn.Module):
    """
        High-level wrapper for different downsampling mechanisms (also called set abstraction mechanisms).
        In general the point cloud is subsampled to produce a lower cardinality point cloud (usualy using farthest point
        sampling (FPS) ). Around each of the resulting points (called central points here) a local neighborhood is
        formed, from which features are aggregated. How features are aggregated can differ, usually this is based on
        maxpooling. This work introduces an attention based alternative.

        Attributes:
            npoint: desired number of points for outpout point cloud
            nneigh: size of neighborhood
            dim: number of dimensions of input and interal dimensions
            type: decides which method to use, options are 'attentive' and 'maxpool'
        """
    def __init__(self, npoint, nneighbor, dim, type='attentive') -> None:
        super().__init__()
        if type == 'attentive':
            self.sa = TransformerSetAbstraction(npoint, nneighbor, dim)
        elif type == 'maxpool':
            self.sa = PointNetSetAbstraction(npoint, nneighbor, dim, dim)
        else:
            raise ValueError('Set Abstraction type ' + type + ' unknown!')

    def forward(self, xyz, feats):
        """
        Executes the downsampling (set abstraction)
        :param xyz: positions of points
        :param feats: features of points
        :return: downsampled version, tuple of (xyz_new, feats_new)
        """
        ret = self.sa(xyz, feats)
        return ret


class TransformerBlock(nn.Module):
    """
    Module for local and global vector self attention, as proposed in the Point Transformer paper.

    Attributes:
        d_model (int): number of input, output and internal dimensions
        k (int): number of points among which local attention is calculated
        pos_only (bool): When set to True only positional features are used
        group_all (bool): When true full instead of local attention is calculated
    """
    def __init__(self, d_model, k, pos_only=False, group_all=False) -> None:
        super().__init__()

        self.pos_only = pos_only

        self.bn = nn.BatchNorm1d(d_model)

        self.fc_delta = nn.Sequential(
            nn.Linear(3, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model)
        )

        self.fc_gamma = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model)
        )
        self.w_qs = nn.Linear(d_model, d_model, bias=False)
        self.w_ks = nn.Linear(d_model, d_model, bias=False)
        self.w_vs = nn.Linear(d_model, d_model, bias=False)
        self.k = k
        self.group_all = group_all

    def forward(self, xyz, feats=None):
        """
        :param xyz [b x n x 3]: positions in point cloud
        :param feats [b x n x d]: features in point cloud
        :return:
            new_features [b x n x d]:
        """

        with torch.no_grad():
            # full attention
            if self.group_all:
                b, n, _ = xyz.shape
                knn_idx = torch.arange(n, device=xyz.device).unsqueeze(0).unsqueeze(1).repeat(b, n, 1)
            # local attention using KNN
            else:
                dists = square_distance(xyz, xyz)
                knn_idx = dists.argsort()[:, :, :self.k]  # b x n x k

        knn_xyz = index_points(xyz, knn_idx)

        if not self.pos_only:
            ori_feats = feats
            x = feats

            q_attn = self.w_qs(x)
            k_attn = index_points(self.w_ks(x), knn_idx)
            v_attn = index_points(self.w_vs(x), knn_idx)

        pos_encode = self.fc_delta(xyz[:, :, None] - knn_xyz)  # b x n x k x d

        if not self.pos_only:
            attn = self.fc_gamma(q_attn[:, :, None] - k_attn + pos_encode)
        else:
            attn = self.fc_gamma(pos_encode)


        attn = functional.softmax(attn, dim=-2)  # b x n x k x d
        if not self.pos_only:
            res = torch.einsum('bmnf,bmnf->bmf', attn, v_attn + pos_encode)
        else:
            res = torch.einsum('bmnf,bmnf->bmf', attn, pos_encode)



        if not self.pos_only:
            res = res + ori_feats
        res = self.bn(res.permute(0, 2, 1)).permute(0, 2, 1)

        return res


class ElementwiseMLP(nn.Module):
    """
    Simple MLP, consisting of two linear layers, a skip connection and batch norm.
    More specifically: linear -> BN -> ReLU -> linear -> BN -> ReLU -> resCon -> BN

    Sorry for that many norm layers. I'm sure not all are needed!
    At some point it was just too late to change it to something proper!
    """
    def __init__(self, dim):
        super().__init__()
        self.conv1 = nn.Conv1d(dim, dim, 1)
        self.bn1 = nn.BatchNorm1d(dim)
        self.conv2 = nn.Conv1d(dim, dim, 1)
        self.bn2 = nn.BatchNorm1d(dim)
        self.bn3 = nn.BatchNorm1d(dim)

    def forward(self, x):
        """
        :param x: [B x n x d]
        :return: [B x n x d]
        """
        x = x.permute(0, 2, 1)
        return self.bn3(x + F.relu(self.bn2(self.conv2(F.relu(self.bn1(self.conv1(x))))))).permute(0, 2, 1)


class PointNetSetAbstraction(nn.Module):
    """
    Set Abstraction Module, as used in PointNet++
    Uses FPS for downsampling, kNN groupings and maxpooling to abstract the group/neighborhood

    Attributes:
        npoint (int): Output cardinality
        nneigh (int): Size of local grouings/neighborhoods
        in_channel (int): input dimensionality
        dim (int): internal and output dimensionality
    """
    def __init__(self, npoint, nneigh, in_channel, dim):
        super(PointNetSetAbstraction, self).__init__()

        self.npoint = npoint
        self.nneigh = nneigh
        self.fc1 = nn.Linear(in_channel, dim)
        self.conv1 = nn.Conv1d(dim, dim, 1)
        self.conv2 = nn.Conv1d(dim, dim, 1)
        self.bn1 = nn.BatchNorm1d(dim)
        self.bn2 = nn.BatchNorm1d(dim)

        self.bn = nn.BatchNorm1d(dim)

    def forward(self, xyz, points):
        """
        Input:
            xyz: input points position data, [B, N, C]
            points: input points data, [B, N, C]
        Return:
            new_xyz: sampled points position data, [B, S, C]
            new_points_concat: sample points feature data, [B, S, D']
        """

        with torch.no_grad():
            fps_idx = pointnet2_utils.furthest_point_sample(xyz, self.npoint).long()

        new_xyz = index_points(xyz, fps_idx)
        points = self.fc1(points)
        points_ori = index_points(points, fps_idx)

        points = points.permute(0, 2, 1)
        points = points + F.relu(self.bn2(self.conv2(F.relu(self.bn1(self.conv1(points))))))
        points = points.permute(0, 2, 1)

        with torch.no_grad():
            dists = square_distance(new_xyz, xyz)  # B x npoint x N
            idx = dists.argsort()[:, :, :self.nneigh]  # B x npoint x K


        grouped_points = index_points(points, idx)


        new_points = points_ori + torch.max(grouped_points, 2)[0]
        new_points = self.bn(new_points.permute(0, 2, 1)).permute(0, 2, 1)
        return new_xyz, new_points


#TODO: can I share some code with PTB??
class TransformerSetAbstraction(nn.Module):
    """
    Newly proposed attention based set abstraction module.
    Uses cross attention from central point to its neighbors instead of maxpooling.

    Attributes:
        npoint (int): Output cardinality of point cloud
        nneigh (int): size of neighborhoods
        dim (int): input, internal and output dimensionality
    """
    def __init__(self, npoint, nneigh, dim):
        super(TransformerSetAbstraction, self).__init__()
        self.npoint = npoint
        self.nneigh = nneigh

        self.bnorm0 = nn.BatchNorm1d(dim)
        self.bnorm1 = nn.BatchNorm1d(dim)
        self.bnorm2 = nn.BatchNorm1d(dim)

        self.bn1 = nn.BatchNorm1d(dim)

        self.conv1 = nn.Conv1d(dim, dim, 1)
        self.conv2 = nn.Conv1d(dim, dim, 1)

        self.fc_delta1 = nn.Sequential(
            nn.Linear(3, dim),
            nn.ReLU(),
            nn.Linear(dim, dim)
        )
        self.fc_gamma1 = nn.Sequential(
            nn.Linear(dim, dim),
            nn.ReLU(),
            nn.Linear(dim, dim)
        )

        self.fc_gamma2 = nn.Sequential(
            nn.Linear(dim, dim),
            nn.ReLU(),
            nn.Linear(dim, dim)
        )

        self.w_qs = nn.Linear(dim, dim, bias=False)
        self.w_ks = nn.Linear(dim, dim, bias=False)
        self.w_vs = nn.Linear(dim, dim, bias=False)

        self.w_qs2 = nn.Linear(dim, dim, bias=False)
        self.w_ks2 = nn.Linear(dim, dim, bias=False)
        self.w_vs2 = nn.Linear(dim, dim, bias=False)

    def forward(self, xyz, points):
        """
        Input: featureized point clouds of cardinality N
            xyz: input points position data, [B, N, 3]
            points: input points data, [B, N, dim]
        Return: downsampled point cloud of cardinality npoint
            new_xyz: sampled points position data, [B, npoint, 3]
            new_points_concat: sample points feature data, [B, npoint, dim]
        """

        B, N, C = xyz.shape

        with torch.no_grad():
            fps_idx = pointnet2_utils.furthest_point_sample(xyz, self.npoint)
            # fps_idx = farthest_point_sample(xyz, self.npoint)
            new_xyz = index_points(xyz, fps_idx.long())
        with torch.no_grad():
            dists = square_distance(new_xyz, xyz)  # B x npoint x N
            idx = dists.argsort()[:, :, :self.nneigh]  # B x npoint x K

        q_attn = index_points(self.w_qs(points), fps_idx.long())
        k_attn = index_points(self.w_ks(points), idx)
        v_attn = index_points(self.w_vs(points), idx)
        grouped_xyz = index_points(xyz, idx)

        pos_encode = self.fc_delta1(grouped_xyz - new_xyz.view(B, self.npoint, 1, C))  # b x n x k x f
        attn = self.fc_gamma1(q_attn[:, :, None] - k_attn + pos_encode)
        attn = functional.softmax(attn, dim=-2)  # b x n x k x f
        res1 = torch.einsum('bmnf,bmnf->bmf', attn, v_attn + pos_encode)

        res1 = res1 + self.conv2(F.relu(self.bn1(self.conv1(res1.permute(0, 2, 1))))).permute(0, 2, 1)
        res1 = self.bnorm0(res1.permute(0, 2, 1)).permute(0, 2, 1)

        q_attn = self.w_qs2(res1)
        k_attn = index_points(self.w_ks2(points), idx)
        v_attn = index_points(self.w_vs2(points), idx)
        attn = self.fc_gamma2(q_attn[:, :, None] - k_attn + pos_encode)
        attn = functional.softmax(attn, dim=-2)  # b x n x k x f
        res2 = torch.einsum('bmnf,bmnf->bmf', attn, v_attn + pos_encode)

        new_points = self.bnorm1((res1 + res2).permute(0, 2, 1)).permute(0, 2, 1)
        new_points = new_points + index_points(points, fps_idx.long())
        new_points = self.bnorm2(new_points.permute(0, 2, 1)).permute(0, 2, 1)

        return new_xyz, new_points