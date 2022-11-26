import torch
import torch.nn as nn
import torch.nn.functional as F
from model.decoder.blocks import CrossTransformerBlock, ResnetBlockFC

class CrossTransformerDecoder(nn.Module):
    """
    AIR-Net decoder

    Attributes:
        dim_inp int: dimensionality of encoding (global and local latent vectors)
        dim int: internal dimensionality
        nneigh int: number of nearest anchor points to draw information from
        hidden_dim int: hidden dimensionality of final feed-forward network
        n_blocks int: number of blocks in feed forward network
    
    Default values:
        dim_inp: 256
        dim: 200
        nneigh: 7
        hidden_dim: 128
        out_dim: 3
    """
    def __init__(self, dim_inp, dim, nneigh=7, hidden_dim=64, n_blocks=5, out_dim=1):
        super().__init__()
        self.dim = dim
        self.n_blocks = n_blocks

        self.ct1 = CrossTransformerBlock(dim_inp, dim, nneigh=nneigh)

        self.init_enc = nn.Linear(dim, hidden_dim)

        self.blocks = nn.ModuleList([
            ResnetBlockFC(hidden_dim) for i in range(n_blocks)
        ])

        self.fc_c = nn.ModuleList([
            nn.Linear(dim, hidden_dim) for i in range(n_blocks)
        ])

        self.fc_out = nn.Linear(hidden_dim, out_dim)
        
        self.actvn = F.relu

    def forward(self, xyz_q, encoding):
        """
        TODO update commont to include encoding dict
        :param xyz_q [B x n_queries x 3]: queried 3D coordinates
        :param lat_rep [B x dim_inp]: global latent vectors
        :param xyz [B x n_anchors x 3]: anchor positions
        :param feats [B x n_anchros x dim_inp]: local latent vectors
        :return: occ [B x n_queries x 3]: deformation vector prediction
        """

        lat_rep = encoding['z']
        xyz = encoding['anchors']
        feats = encoding['anchor_feats']
        
        # lobal_field:
        #    lat_rep = self.fc_glob(lat_rep).unsqueeze(1).repeat(1, xyz_q.shape[1], 1)
        #    net = self.fc_p(xyz_q)
        lat_rep = self.ct1(xyz_q, lat_rep, xyz, feats)  
        net = self.init_enc(lat_rep)

        for i in range(self.n_blocks):
            net = net + self.fc_c[i](lat_rep)
            net = self.blocks[i](net)

        occ = self.fc_out(self.actvn(net))        
        return occ