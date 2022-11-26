
import torch
import torch.nn as nn
import torch.nn.functional as F
from model.decoder.blocks import ResnetBlockFC


class PointInterpDecoder(nn.Module):
    """
    Decoder based in interpolation features between local latent vectors.
    Gaussian Kernel regression is used for the interpolation of features.
    Coda adapted from https://github.com/autonomousvision/convolutional_occupancy_networks

    Attributes:
        dim_inp: input dimensionality
        hidden_dim: dimensionality for feed-forward network
        n_blocks: number of blocks in feed worward network
        var (float): variance for gaussian kernel
    
    Default Values:
        dim_inp: 256
        dim: 200 
        hidden_dim: 128, 
        out_dim: 3
    """
    def __init__(self, dim_inp, dim, out_dim=3, hidden_dim=50, n_blocks=5):
        super().__init__()
        self.n_blocks = n_blocks

        self.fc0 = nn.Linear(dim_inp, dim)

        self.fc1 = nn.Linear(dim, hidden_dim)


        self.blocks = nn.ModuleList([
            ResnetBlockFC(hidden_dim) for i in range(n_blocks)
        ])

        self.fc_c = nn.ModuleList([
            nn.Linear(dim, hidden_dim) for i in range(n_blocks)
        ])

        self.fc_out = nn.Linear(hidden_dim, out_dim)

        self.actvn = F.relu

        self.var = 0.2**2


    def sample_point_feature(self, q, p, fea):
        # q: B x M x 3
        # p: B x N x 3
        # fea: B x N x c_dim
        # p, fea = c

        # distance betweeen each query point to the point cloud
        dist = -((p.unsqueeze(1).expand(-1, q.size(1), -1, -1) - q.unsqueeze(2)).norm(dim=3) + 10e-6) ** 2
        weight = (dist / self.var).exp()  # Guassian kernel

        # weight normalization
        weight = weight / weight.sum(dim=2).unsqueeze(-1)

        c_out = weight @ fea  # B x M x c_dim

        return c_out


    def forward(self, xyz_q, encoding):
        """
        :param xyz_q [B x n_quries x 3]: queried 3D positions
        :param xyz [B x n_anchors x 3]: anchor positions
        :param feats [B x n_anchors x dim_inp]: anchor features
        :return: occ [B x n_queries x 3]: deformation predictions
        """

        xyz = encoding['anchors']
        feats = encoding['anchor_feats']

        lat_rep = self.fc0(self.sample_point_feature(xyz_q, xyz, feats))

        net = self.fc1(F.relu(lat_rep))

        for i in range(self.n_blocks):
            net = net + self.fc_c[i](lat_rep)
            net = self.blocks[i](net)

        occ = self.fc_out(self.actvn(net))
        return occ
