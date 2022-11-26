import torch
import torch.nn as nn
import torch.nn.functional as functional
import numpy as np
from time import time
import torch.nn.functional as F

from model.encoder import encoder_dict
from model.decoder import decoder_dict
from model.utils import compute_l2_error

class Deformation_Networks(nn.Module):
    def __init__(self, cfg, no_input_corr=False):

        super(Deformation_Networks, self).__init__()
        self.no_input_corr = no_input_corr
        if self.no_input_corr:
            if cfg['model']['use_normals']:
                has_features=True
                inp_feat_dim=3
            else:
                has_features=False
                inp_feat_dim=0
        else:
            if cfg['model']['use_normals']:
                has_features=True
                inp_feat_dim=7
            else:
                has_features=True
                inp_feat_dim=4
                
        encoder        = cfg['model']['encoder']
        encoder_kwargs = cfg['model']['encoder_kwargs']
        self.encoder = encoder_dict[encoder](
                                has_features=has_features, inp_feat_dim=inp_feat_dim,
                                **encoder_kwargs,
                        )
            
        decoder        = cfg['model']['decoder']
        decoder_kwargs = cfg['model']['decoder_kwargs']
        self.decoder = decoder_dict[decoder](**decoder_kwargs)

    def forward(self, points, surface_samples_inputs):
        batch_size = points.shape[0]
        num_points = points.shape[1]
        
        ################################################################################################
        # Geometry & Flow encoding.
        ################################################################################################     
        if self.no_input_corr:
            encoding = self.encoder(surface_samples_inputs[:, :, 0:3].contiguous())
        else:
            encoding = self.encoder(surface_samples_inputs)
        
        ################################################################################################
        # Flow decoding.
        ################################################################################################
        deformed_points = self.decoder(points, encoding)
        
        return deformed_points
    
    
def train_on_batch_with_cano(model, optimizer, data_dict, config):
    # Make sure that everything has the correct size
    optimizer.zero_grad()
    surface_samples_inputs = data_dict['surface_samples_inputs']
    source_points = data_dict['space_samples_src']
    target_points = data_dict['space_samples_tgt']
    deformed_points = model(source_points, surface_samples_inputs)
    # Compute the loss
    loss = compute_l2_error(deformed_points, target_points)
    # Do the backpropagation
    loss.backward()
    # Do the update
    optimizer.step()

    return loss.item()


@torch.no_grad()
def validate_on_batch_with_cano(model, data_dict, config):
    surface_samples_inputs = data_dict['surface_samples_inputs']
    source_points = data_dict['space_samples_src']
    target_points = data_dict['space_samples_tgt']
    deformed_points = model(source_points, surface_samples_inputs)
    # Compute the loss
    loss = compute_l2_error(deformed_points, target_points)
    return loss.item()

@torch.no_grad()
def test_on_batch_with_cano(model, data_dict, config, compute_loss=False):
    surface_samples_inputs = data_dict['surface_samples_inputs']
    source_points = data_dict['surface_samples_src']
    target_points = data_dict['surface_samples_tgt']
    
    deformed_points = model(source_points, surface_samples_inputs)
    data_dict['surface_samples_tgt_pred'] = deformed_points
    
    source_verts = data_dict['verts_src']
    target_verts = data_dict['verts_tgt']
    deformed_verts = model(source_verts, surface_samples_inputs)
    data_dict['verts_tgt_pred'] = deformed_verts
    
    # Compute the loss
    if compute_loss:
        loss = compute_l2_error(deformed_verts, target_verts)
    else:
        loss = torch.zeros((1), dtype=torch.float32)
    return loss.item(), data_dict