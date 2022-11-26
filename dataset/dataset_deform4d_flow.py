import pstats
import sys,os

import math
from tkinter import E
import torch
import numpy as np
from skimage import io, transform
from torch.utils.data import Dataset, dataloader
import json
import open3d as o3d
import pickle
import random
import struct
from timeit import default_timer as timer
from tqdm import tqdm
import trimesh
from scipy.spatial import KDTree
import dataset.utils as data_utils

class Deform4DFlow_Dataset(Dataset):
    def __init__(self, cfg, iden_split, motion_split, load_mesh=False, num_sampled_pairs=-1):
        self.cfg                 = cfg
        self.iden_split          = iden_split
        self.motion_split        = motion_split
        self.load_mesh           = load_mesh
        self.num_sampled_pairs   = num_sampled_pairs
        self.dataset_type        = cfg['data']['type'] 
        self.dataset_dir         = cfg['data']['dataset_dir']   
        self.split_dir           = cfg['data']['split_dir']
     
        # Load deform pairs information
        self.all_deform_pairs = []
        self.sample_deform_pairs = []
        self._load()

    def _load(self):
        # read identity models used as canonical meshes
        # we use the first frame e.g. 0-th frame as the canonical pose of the identity models
        split_file = os.path.join(self.split_dir, self.dataset_type, self.iden_split + '.lst')
        with open(split_file, 'r') as f:
            iden_seq_names = f.read().split('\n')          
        iden_seq_dirs = sorted([os.path.join(self.dataset_dir, iden_seq_name) for iden_seq_name in iden_seq_names if os.path.isdir(os.path.join(self.dataset_dir, iden_seq_name)) and iden_seq_name !=''])

        self.models_cano_dict = {}  # used to save the information of each identityname
        for idx_cano in range(len(iden_seq_dirs)):
            cano_seq_dir = iden_seq_dirs[idx_cano]
            cano_seq_name = os.path.basename(cano_seq_dir)
            # in the deformingthing4d, the deforming/animation sequence  is named as 'identityname_motionname'
            iden_name = cano_seq_name.split('_')[0]
            self.models_cano_dict[iden_name] = (idx_cano, cano_seq_name)  

        #read temporal animations / sequences
        split_file = os.path.join(self.split_dir, self.dataset_type, self.motion_split + '.lst')
        with open(split_file, 'r') as f:
            motion_seq_names = f.read().split('\n')          
        motion_seq_dirs = sorted([os.path.join(self.dataset_dir, motion_seq_name) for motion_seq_name in motion_seq_names if os.path.isdir(os.path.join(self.dataset_dir, motion_seq_name)) and motion_seq_name !=''])
        
        self.models_motion_dict = {} # used to save the information of each sequence
        for idx_motion in range(len(motion_seq_dirs)):
            motion_seq_dir = motion_seq_dirs[idx_motion]
            motion_seq_name = os.path.basename(motion_seq_dir)
            # in the deformingthing4d, the deforming/animation sequence is named as 'identityname_motionname'
            self.models_motion_dict[motion_seq_name] = (idx_motion, motion_seq_name)  
            
        
        # form deformation pairs between non-rigid poses and their corresponding canonical poses
        self.all_deform_pairs = []
        for motion_seq_name in motion_seq_names:
            if os.path.isdir(os.path.join(self.dataset_dir, motion_seq_name)) and motion_seq_name !='':
                cano_name = motion_seq_name.split('_')[0]
                if motion_seq_name not in self.models_motion_dict or cano_name not in self.models_cano_dict:
                    continue
                idx_cano, cano_seq_name = self.models_cano_dict[cano_name]  
                idx_motion, _ = self.models_motion_dict[motion_seq_name]  

                frame_names = sorted(os.listdir(os.path.join(self.dataset_dir, motion_seq_name)))
                frame_names = [frame_name for frame_name in frame_names if int(frame_name) % self.cfg['data']['interval']==0 ]

                if self.cfg['data']['arbitrary']:
                    # for shape transformations between two arbitrary poses:
                    # i.e. the end-to-end finetune stage of the whole network
                    # train: randomly sample two poses from a temporal sequence as source and target mesh
                    if self.motion_split[:5] == "train":
                        for frame_name0 in frame_names:
                            for frame_name1 in frame_names:
                                cano_seq_frame_name     = "0000"
                                motion_seq_frame_name0  = frame_name0
                                motion_seq_frame_name1  = frame_name1
                                self.all_deform_pairs.append({
                                    "pair_info": (idx_cano, cano_seq_name, cano_seq_frame_name,  
                                                idx_motion, motion_seq_name, motion_seq_frame_name0, motion_seq_name, motion_seq_frame_name1)
                                    })
                    # used in the qualitative and quantitative comparions
                    # val / test: use the first frame as the source mesh, use the randomly sample frame as the target mesh
                    else:
                        for frame_name in frame_names:
                            if int(frame_name)>0:
                                cano_seq_frame_name     = "0000"
                                motion_seq_frame_name0  = '0000' 
                                motion_seq_frame_name1  = frame_name
                                self.all_deform_pairs.append({
                                    "pair_info": (idx_cano, cano_seq_name, cano_seq_frame_name,  
                                        idx_motion, motion_seq_name, motion_seq_frame_name0, motion_seq_name, motion_seq_frame_name1)
                                    })
                else:
                    # for shape transformations between canonical pose and other poses, 
                    # i.e. the pretraining stage of forward and backward deformation networks
                    # source mesh: the first frame of the canonical mesh sequence
                    # target mesh: the randomly sample frame of a non-rigidly sampled mesh sequence
                    for frame_name in frame_names:
                        cano_seq_frame_name  = '0000' 
                        motion_seq_frame_name = frame_name
                        self.all_deform_pairs.append({
                                "pair_info": (idx_cano, cano_seq_name, cano_seq_frame_name, 
                                    idx_motion, cano_seq_name, cano_seq_frame_name, motion_seq_name, motion_seq_frame_name)
                                })

        if self.motion_split[:5] == "train" or self.num_sampled_pairs > 0:
            self.random_shuffle_samples(self.num_sampled_pairs)
        else:
            self.sample_deform_pairs = self.all_deform_pairs
    
    def random_shuffle_samples(self, num_samples=-1): 
        random.Random(100).shuffle(self.all_deform_pairs)
        if num_samples > 0:
            self.sample_deform_pairs = self.all_deform_pairs[:num_samples]
        else:
            self.sample_deform_pairs = self.all_deform_pairs

    def __len__(self):
        return len(self.sample_deform_pairs)

    def get_metadata(self, index):
        return self.sample_deform_pairs[index]

    def _load_data(self, data_dir):
        data_dict = {}
        # Load data from directory.
        norm_params_filepath = os.path.join(data_dir, self.cfg['data']['norm_params_file'])
        orig2world = np.reshape(np.loadtxt(norm_params_filepath), [4, 4]).astype(np.float32)
        world2orig = np.linalg.inv(orig2world).astype(np.float32)
        surface_flow_filepath = os.path.join(data_dir, self.cfg['data']['surface_flow_file'] )
        surface_flow_samples, surface_flow_normals = data_utils.load_npz_surface_flow(surface_flow_filepath)
        space_flow_filepath = os.path.join(data_dir, self.cfg['data']['space_flow_file']) 
        space_flow_samples = data_utils.load_npz_space_flow(space_flow_filepath)
        if self.cfg['data']['fix_coord_system'] :
            surface_flow_samples = data_utils.fix_coord_system(surface_flow_samples) 
            surface_flow_normals = data_utils.fix_coord_system(surface_flow_normals)
            space_flow_samples   = data_utils.fix_coord_system(space_flow_samples)
        data_dict = {
                'orig2world':               orig2world,
                'world2orig':               world2orig,
                'surface_samples':     surface_flow_samples,
                'surface_normals':     surface_flow_normals,
                'space_samples':       space_flow_samples,
            }

        if self.load_mesh:
            mesh_filepath = os.path.join(data_dir, self.cfg['data']['mesh_file'] )
            verts, edges, faces = data_utils.load_mesh_info(mesh_filepath)
            # normalization
            if "norm" not in self.cfg['data']['mesh_file']:
                verts  = (np.matmul(orig2world[:3, :3], verts.T) + orig2world[:3, 3:4]).T
            # fix coord system
            if self.cfg['data']['fix_coord_system'] :
                verts = data_utils.fix_coord_system(verts)
            data_dict['verts'] = verts
            data_dict['edges'] = edges
            data_dict['faces'] = faces      

        return data_dict

    def __getitem__(self, index):
        out_dict = {}
        
        # get the index-th pair_info of  sample_deform_pairs
        idx_cano, cano_seq_name, cano_seq_frame_name, \
            idx_motion, source_seq_name, source_seq_frame_name, target_seq_name, target_seq_frame_name \
                = self.sample_deform_pairs[index]["pair_info"]

        # when mode == "train" and index is the last data sample:
        if self.motion_split[:5] == "train" and index == len(self.sample_deform_pairs)-1:
            self.random_shuffle_samples(self.num_sampled_pairs)
            print("random shuffle deformation pairs in train dataset, and sample again")
        
        # get the cano/src/tgt mesh directories and data
        data_dir_cano = os.path.join(self.dataset_dir, cano_seq_name,   cano_seq_frame_name)  
        data_dir_src  = os.path.join(self.dataset_dir, source_seq_name, source_seq_frame_name)
        data_dir_tgt  = os.path.join(self.dataset_dir, target_seq_name, target_seq_frame_name)
        data_cano = self._load_data(data_dir_cano)
        if not self.cfg['data']['arbitrary']:
            if self.cfg['data']['inverse']:
                # from arbitraty pose to canonical pose -- used when we train backward deformation network
                data_src  = self._load_data(data_dir_tgt)
                data_tgt  = self._load_data(data_dir_src)
                # from canonical pose to arbitraty pose-- used when we train forward deformation network
            else:
                data_src  = self._load_data(data_dir_src)
                data_tgt  = self._load_data(data_dir_tgt)
        else:
            data_src  = self._load_data(data_dir_src)
            data_tgt  = self._load_data(data_dir_tgt)
        
        # Subsample surface flow samples & normals
        surface_samples_cano, surface_normals_cano  = data_cano['surface_samples'], data_cano['surface_normals']
        surface_samples_src,  surface_normals_src   = data_src['surface_samples'],  data_src['surface_normals']
        surface_samples_tgt,  surface_normals_tgt   = data_tgt['surface_samples'],  data_tgt['surface_normals']
        surface_samples_cano_bbox_min, surface_samples_cano_bbox_max = surface_samples_cano.min(axis=0), surface_samples_cano.max(axis=0)
        surface_samples_cano, surface_samples_src, surface_samples_tgt, surface_samples_idxs = \
            data_utils.subsample_surface_flow( self.cfg, surface_samples_cano, surface_samples_src, surface_samples_tgt )
        surface_normals_cano, surface_normals_src, surface_normals_tgt, _ = \
            data_utils.subsample_surface_flow( self.cfg,  surface_normals_cano, surface_normals_src, surface_normals_tgt, surface_samples_idxs=surface_samples_idxs)
        
        # compute handle mask & mask sample flow
        cano_handle_sample_idx = data_utils.cano_sample_handle_mask( self.cfg, surface_samples_cano, surface_samples_cano_bbox_min, surface_samples_cano_bbox_max )
        surface_samples_tgt_masked = surface_samples_tgt * cano_handle_sample_idx[:, None]
        # add noise to the source mesh
        if self.cfg['data']['noise_level'] > 0.0:
            surface_samples_src = data_utils.add_noise_to_src( self.cfg, surface_samples_src)
        # concat src surface samples & masked flow & handle mask to form surface_samples_inputs 
        surface_samples_inputs = np.concatenate([surface_samples_src, surface_samples_tgt_masked, cano_handle_sample_idx[:, None]], axis=1)
        surface_samples_inputs = surface_samples_inputs.astype(np.float32)
        # if create partial shape, we auto complete the surface_samples to a fix number of points
        if self.cfg['data']['partial_shape_ratio'] < 1.0:
            remain_idx = data_utils.create_partial_src( self.cfg, surface_samples_src, cano_handle_sample_idx)
            surface_samples_inputs = surface_samples_inputs[:, remain_idx, :]
            surface_samples_cano, surface_samples_src, surface_samples_tgt = surface_samples_cano[remain_idx], surface_samples_src[remain_idx], surface_samples_tgt[remain_idx]
            surface_normals_cano, surface_normals_src, surface_normals_tgt = surface_normals_cano[remain_idx], surface_normals_src[remain_idx], surface_normals_tgt[remain_idx]
            cano_handle_sample_idx = cano_handle_sample_idx[remain_idx]
        out_dict['surface_samples_cano'], out_dict['surface_samples_src'], out_dict['surface_samples_tgt'] = surface_samples_cano, surface_samples_src, surface_samples_tgt 
        out_dict['surface_normals_cano'], out_dict['surface_normals_src'], out_dict['surface_normals_tgt'] = surface_normals_cano, surface_normals_src, surface_normals_tgt
        out_dict['cano_handle_sample_idx'] = cano_handle_sample_idx[:, None]
        out_dict['surface_samples_inputs'] = surface_samples_inputs
        
        # subsample space flow samples
        space_samples_cano = data_cano['space_samples']
        space_samples_src  = data_src['space_samples']
        space_samples_tgt  = data_tgt['space_samples']
        space_samples_cano, space_samples_src, space_samples_tgt = \
            data_utils.subsample_space_flow( self.cfg, space_samples_cano, space_samples_src, space_samples_tgt)
        out_dict['space_samples_cano'], out_dict['space_samples_src'], out_dict['space_samples_tgt'] = space_samples_cano, space_samples_src, space_samples_tgt
        
        if self.load_mesh:
            verts_cano = data_cano['verts']
            verts_src  = data_src['verts']
            verts_tgt  = data_tgt['verts']
            edges = data_cano['edges']
            faces = data_cano['faces']

            # mask vertices flow
            cano_vert_bbox_min, cano_vert_bbox_max = verts_cano.min(axis=0), verts_cano.max(axis=0) 
            cano_handle_vert_idx = data_utils.cano_vert_handle_mask(self.cfg, verts_cano, cano_vert_bbox_min, cano_vert_bbox_max)
            verts_tgt_masked = verts_tgt * cano_handle_vert_idx[:, None]
            # concat src surface verts & masked verts flow & handle verts mask to form surface_verts_input
            verts_flow_inputs = np.concatenate([verts_src, verts_tgt_masked, cano_handle_vert_idx[:, None]], axis=1)
            out_dict['verts_cano'],  out_dict['verts_src'],  out_dict['verts_tgt'] = verts_cano, verts_src, verts_tgt
            out_dict['cano_handle_vert_idx'] = cano_handle_vert_idx[:, None]
            out_dict['verts_flow_inputs'] = verts_flow_inputs
            out_dict['edges'] = edges
            out_dict['faces'] = faces

        out_dict['index'] = index
        return out_dict

    @staticmethod
    def collate_fn(samples):
        ''' Collater that puts each data field into a tensor with outer dimension
            batch size.
        Args:
            samples: samples
        '''
    
        samples = list(filter(lambda x: x is not None, samples))
        return dataloader.default_collate(samples)