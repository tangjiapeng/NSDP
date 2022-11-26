import torch
import torch.nn as nn
import os
from model.encoder.blocks import TransitionDown, ElementwiseMLP, TransformerBlock

class PointTransformerEncoder(nn.Module):
    """
    Attributes:
        npoints_per_layer [int]: cardinalities of point clouds for each layer
        nneighbor int: number of neighbors in local vector attention (also in TransformerSetAbstraction)
        nneighbor_reduced int: number of neighbors in very first TransformerBlock
        nfinal_transformers int: number of full attention layers
        d_transformer int: dimensionality of model
        d_reduced int: dimensionality of very first layers
        full_SA bool: if False, local self attention is used in final layers
        has_features bool: True, when input has signed-distance value for each point

        default value:
            npoints_per_layer: [5000, 500, 100]
            nneighbor: 16
            nneighbor_reduced: 10
            nfinal_transformers: 3
            d_transformer: 256
            d_reduced: 120
            full_SA: true
    """
    def __init__(self, npoints_per_layer, nneighbor, nneighbor_reduced, nfinal_transformers,
                 d_transformer, d_reduced,
                 full_SA=False, has_features=False, inp_feat_dim=1):
        super().__init__()
        self.d_reduced = d_reduced
        self.d_transformer = d_transformer
        self.has_features = has_features

        self.fc_middle = nn.Sequential(
            nn.Linear(d_transformer, d_transformer),
            nn.ReLU(),
            nn.Linear(d_transformer, d_transformer)
        )
        if self.has_features:
            self.enc_sdf = nn.Linear(inp_feat_dim, d_reduced)
        self.transformer_begin = TransformerBlock(d_reduced, nneighbor_reduced,
                                                  pos_only=not self.has_features)
        self.transition_downs = nn.ModuleList()
        self.transformer_downs = nn.ModuleList()
        self.elementwise = nn.ModuleList()
        #self.transformer_downs2 = nn.ModuleList() #compensate
        #self.elementwise2 = nn.ModuleList() # compensate
        self.elementwise_extras = nn.ModuleList()

        if not d_reduced == d_transformer:
            self.fc1 = nn.Linear(d_reduced, d_transformer)

        for i in range(len(npoints_per_layer) - 1):
            old_npoints = npoints_per_layer[i]
            new_npoints = npoints_per_layer[i + 1]

            if i == 0:
                dim = d_reduced
            else:
                dim = d_transformer
            self.transition_downs.append(
                TransitionDown(new_npoints, min(nneighbor, old_npoints), dim) # , type='single_step')  #, type='maxpool')#, type='single_step')
            )
            self.elementwise_extras.append(ElementwiseMLP(dim))
            self.transformer_downs.append(
                TransformerBlock(dim, min(nneighbor, new_npoints))
            )
            self.elementwise.append(ElementwiseMLP(d_transformer))
            #self.transformer_downs2.append(
            #    TransformerBlock(dim, min(nneighbor, new_npoints))
            #) # compensate
            #self.elementwise2.append(ElementwiseMLP(dim)) # compensate

        self.final_transformers = nn.ModuleList()
        self.final_elementwise = nn.ModuleList()

        for i in range(nfinal_transformers):
            self.final_transformers.append(
                TransformerBlock(d_transformer, 2 * nneighbor, group_all=full_SA)
            )
        for i in range(nfinal_transformers):
            self.final_elementwise.append(
                ElementwiseMLP(dim=d_transformer)
            )

    def forward(self, xyz, intermediate_out_path=None):
        """
        :param xyz [B x n x 3] (or [B x n x 4], but then has_features=True): input point cloud
        :param intermediate_out_path: path to store point cloud after every deformation to
        :return: global latent representation [b x d_transformer]
                 xyz [B x npoints_per_layer[-1] x d_transformer]: anchor positions
                 feats [B x npoints_per_layer[-1] x d_transformer: local latent vectors
        """

        if intermediate_out_path is not None:
            intermediates = {}
            intermediates['Input'] = xyz[0, :, :].cpu().numpy()

        if self.has_features:
            feats = self.enc_sdf(xyz[:, :, 3:])
            xyz = xyz[:, :, :3].contiguous()
            feats = self.transformer_begin(xyz, feats)
        else:
            feats = self.transformer_begin(xyz)

        for i in range(len(self.transition_downs)):
            xyz, feats = self.transition_downs[i](xyz, feats)

            if intermediate_out_path is not None:
                intermediates['SetAbs{}'.format(i)] = xyz[0, :, :].cpu().numpy()

            feats = self.elementwise_extras[i](feats)
            feats = self.transformer_downs[i](xyz, feats)
            if intermediate_out_path is not None:
                intermediates['PTB{}'.format(i)] = xyz[0, :, :].cpu().numpy()
            #feats = self.transformer_downs2[i](xyz, feats) #compensate: dense
            #feats = self.elementwise2[i](feats) #compensate: dense
            if i == 0 and not self.d_reduced == self.d_transformer:
                feats = self.fc1(feats)
            feats = self.elementwise[i](feats)
            #feats = self.transformer_downs2[i](xyz, feats) #compensate: sparse
            #feats = self.elementwise2[i](feats) #compensate: sparse

        for i, att_block in enumerate(self.final_transformers):
            feats = att_block(xyz, feats)
            #if i < len(self.final_elementwise):
            feats = self.final_elementwise[i](feats)
            if intermediate_out_path is not None:
                intermediates['fullPTB{}'.format(i)] = xyz[0, :, :].cpu().numpy()

        if intermediate_out_path is not None:
            if not os.path.exists(intermediate_out_path):
                os.makedirs(intermediate_out_path)
            np.savez(intermediate_out_path + '/intermediate_pcs.npz', **intermediates)

        # max pooling
        lat_vec = feats.max(dim=1)[0]

        return {'z': self.fc_middle(lat_vec), 'anchors': xyz, 'anchor_feats': feats}

