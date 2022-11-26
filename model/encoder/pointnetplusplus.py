import torch
import torch.nn as nn
from model.encoder.blocks import TransitionDown, ElementwiseMLP, TransformerBlock

class PointNetPlusPlusEncoder(nn.Module):
    """
    PointNet++-style encoder. Used in ablation experiments.

    Attributes:
        npoints_per_layer [int]: cardinality of point cloud for each layer
        nneighbor int: number of neighbors for set abstraction
        d_transformer int: internal dimensions

    Default values:
        npoints_per_layer: [5000, 500, 100]
        nneighbor: 16
        d_transformer: 256
        nfinal_transformers: 3
    """
    def __init__(self, npoints_per_layer, nneighbor, d_transformer, nfinal_transformers, has_features=False, inp_feat_dim=1):
        super().__init__()
        
        self.d_transformer = d_transformer

        self.fc_middle = nn.Sequential(
            nn.Linear(d_transformer, d_transformer),
            nn.ReLU(),
            nn.Linear(d_transformer, d_transformer)
        )

        ##### add has features / inp_feat_dim args
        self.has_features = has_features
        self.inp_feat_dim = inp_feat_dim
        if has_features:
            self.fc_begin = nn.Sequential(
                nn.Linear(inp_feat_dim, d_transformer),
                nn.ReLU(),
                nn.Linear(d_transformer, d_transformer)
            )
        else:
            self.fc_begin = nn.Sequential(
                nn.Linear(3, d_transformer),
                nn.ReLU(),
                nn.Linear(d_transformer, d_transformer)
            )

        self.transition_downs = nn.ModuleList()
        self.elementwise = nn.ModuleList()

        for i in range(len(npoints_per_layer) - 1):
            old_npoints = npoints_per_layer[i]
            new_npoints = npoints_per_layer[i + 1]
            self.transition_downs.append(
                TransitionDown(new_npoints, min(nneighbor, old_npoints), d_transformer, type='maxpool')
            )
            self.elementwise.append(ElementwiseMLP(d_transformer))

        # full self attention layers
        self.final_transformers = nn.ModuleList()
        self.final_elementwise = nn.ModuleList()

        for i in range(nfinal_transformers):
            self.final_transformers.append(
                TransformerBlock(d_transformer, -1, group_all=True)
            )
        for i in range(nfinal_transformers):
            self.final_elementwise.append(
                ElementwiseMLP(dim=d_transformer)
            )

    def forward(self, xyz):
        """
        :param xyz [B x n x 3] (or [B x n x 4], but then has_features=True): input point cloud
        :param intermediate_out_path: path to store point cloud after every deformation to
        :return: global latent representation [b x d_transformer]
                 xyz [B x npoints_per_layer[-1] x d_transformer]: anchor positions
                 feats [B x npoints_per_layer[-1] x d_transformer: local latent vectors
        """
        if self.has_features:
            feats = self.fc_begin(xyz[:, :, 3:].contiguous())
            xyz   = xyz[:, :, 0:3].contiguous()
        else:
            feats = self.fc_begin(xyz)

        for i in range(len(self.transition_downs)):
            xyz, feats = self.transition_downs[i](xyz, feats)
            feats = self.elementwise[i](feats)

        for i, att_block in enumerate(self.final_transformers):
            feats = att_block(xyz, feats)
            feats = self.final_elementwise[i](feats)

        # max pooling
        lat_vec = feats.max(dim=1)[0]

        return {'z': self.fc_middle(lat_vec), 'anchors': xyz, 'anchor_feats': feats}