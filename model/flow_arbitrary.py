import os
import numpy as np
import torch
import torch.nn as nn
from model.utils import compute_l2_error

class FlowArbitrary(nn.Module):
    def __init__(self, cfg, model_canonicalize, model_deform):
        super(FlowArbitrary, self).__init__()

        self.model_canonicalize = model_canonicalize
        self.model_deform = model_deform
        

    def forward(self, space_samples_src, surface_samples_src, surface_samples_tgt, cano_handle_sample_mask):
        ################################################################################################
        # From source pose to canonical pose.
        ################################################################################################  
        space_samples_src2cano   = self.model_canonicalize(space_samples_src, surface_samples_src)
        surface_samples_src2cano = self.model_canonicalize(surface_samples_src, surface_samples_src)
        
        ################################################################################################
        # From canonical pose to target pose
        ################################################################################################ 
        space_samples_src2tgt = self.model_deform( space_samples_src2cano, torch.cat([surface_samples_src2cano, surface_samples_tgt, cano_handle_sample_mask],dim=-1).contiguous() ) 

        return space_samples_src2tgt
    

def train_on_batch_with_arbitrary(model, optimizer, data_dict, config):
    # Make sure that everything has the correct size
    optimizer.zero_grad()
    surface_samples_inputs = data_dict['surface_samples_inputs']
    surface_samples_src = surface_samples_inputs[:, :, 0:3]
    surface_samples_tgt = surface_samples_inputs[:, :, 3:6]
    cano_handle_sample_mask = surface_samples_inputs[:, :, 6:7]
    
    space_samples_src   = data_dict['space_samples_src']
    space_samples_tgt   = data_dict['space_samples_tgt']
    space_samples_src2tgt = model(space_samples_src, surface_samples_src, surface_samples_tgt, cano_handle_sample_mask)
    # Compute the loss
    loss = compute_l2_error(space_samples_src2tgt, space_samples_tgt)
    # Do the backpropagation
    loss.backward()
    # Do the update
    optimizer.step()

    return loss.item()


@torch.no_grad()
def validate_on_batch_with_arbitrary(model, data_dict, config):
    surface_samples_inputs = data_dict['surface_samples_inputs']
    surface_samples_src = surface_samples_inputs[:, :, 0:3]
    surface_samples_tgt = surface_samples_inputs[:, :, 3:6]
    cano_handle_sample_mask = surface_samples_inputs[:, :, 6:7]
    
    space_samples_src   = data_dict['space_samples_src']
    space_samples_tgt   = data_dict['space_samples_tgt']
    space_samples_src2tgt = model(space_samples_src, surface_samples_src, surface_samples_tgt, cano_handle_sample_mask)
    # Compute the loss
    loss = compute_l2_error(space_samples_src2tgt, space_samples_tgt)
    return loss.item()

@torch.no_grad()
def test_on_batch_with_arbitrary(model, data_dict, config, compute_loss=False):
    surface_samples_inputs = data_dict['surface_samples_inputs']
    surface_samples_src = surface_samples_inputs[:, :, 0:3]
    surface_samples_tgt = surface_samples_inputs[:, :, 3:6]
    cano_handle_sample_mask = surface_samples_inputs[:, :, 6:7]
    
    deformed_samples = model(surface_samples_src, surface_samples_src, surface_samples_tgt, cano_handle_sample_mask)
    data_dict['surface_samples_tgt_pred'] = deformed_samples
    
    source_verts = data_dict['verts_src']
    target_verts = data_dict['verts_tgt']
    deformed_verts = model(source_verts, surface_samples_src, surface_samples_tgt, cano_handle_sample_mask)
    data_dict['verts_tgt_pred'] = deformed_verts

    # Compute the loss
    if compute_loss:
        loss = compute_l2_error(deformed_verts, target_verts)
    else:
        loss = torch.zeros((1), dtype=torch.float32)
    return loss.item(), data_dict