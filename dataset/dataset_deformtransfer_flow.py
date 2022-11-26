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
from dataset.dataset_deform4d_flow import Deform4DFlow_Dataset

class DeformTransferFlow_Dataset(Deform4DFlow_Dataset):
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
        # split_file = os.path.join(self.split_dir, self.dataset_type, self.iden_split + '.lst')
        # with open(split_file, 'r') as f:
        #     iden_seq_names = f.read().split('\n')          
        # iden_seq_dirs = sorted([os.path.join(self.dataset_dir, iden_seq_name) for iden_seq_name in iden_seq_names if os.path.isdir(os.path.join(self.dataset_dir, iden_seq_name)) and iden_seq_name !=''])
        
        # self.models_cano_dict = {}  # used to save the information of each identityname
        # for idx_cano in range(len(iden_seq_dirs)):
        #     cano_seq_dir = iden_seq_dirs[idx_cano]
        #     cano_seq_name = os.path.basename(cano_seq_dir)
        #     # in the deformtransfer, the deforming/animation sequence  is named as 'identityname-motionname'
        #     iden_name = cano_seq_name.split('_')[0]
        #     self.models_cano_dict[iden_name] = (idx_cano, cano_seq_name)  


        #read temporal animations / sequences
        split_file = os.path.join(self.split_dir, self.dataset_type, self.motion_split + '.lst')
        with open(split_file, 'r') as f:
            motion_seq_names = f.read().split('\n')          
        motion_seq_dirs = sorted([os.path.join(self.dataset_dir, motion_seq_name) for motion_seq_name in motion_seq_names if os.path.isdir(os.path.join(self.dataset_dir, motion_seq_name)) and motion_seq_name !=''])
        
        self.models_motion_dict = {} # used to save the information of each sequence
        for idx_motion in range(len(motion_seq_dirs)):
            motion_seq_dir = motion_seq_dirs[idx_motion]
            motion_seq_name = os.path.basename(motion_seq_dir)
            # in the deformtransfer, the deforming/animation sequence  is named as 'identityname-motionname'
            self.models_motion_dict[motion_seq_name] = (idx_motion, motion_seq_name)  
            
        
        # form deformation pairs between non-rigid poses and their corresponding canonical poses
        self.all_deform_pairs = []
        for motion_seq_name in motion_seq_names:
            if os.path.isdir(os.path.join(self.dataset_dir, motion_seq_name)) and motion_seq_name !='':
                # cano_name = motion_seq_name.split('_')[0]
                # if motion_seq_name not in self.models_motion_dict or cano_name not in self.models_cano_dict:
                #     continue
                #idx_cano, cano_seq_name = self.models_cano_dict[cano_name]  
                idx_motion, _ = self.models_motion_dict[motion_seq_name]  

                frame_names = sorted(os.listdir(os.path.join(self.dataset_dir, motion_seq_name)))
                frame_names = [frame_name for frame_name in frame_names if int(frame_name) % self.cfg['data']['interval']==0 ]

                if self.cfg['data']['arbitrary']:
                    # for shape transformations between two arbitrary poses:
                    # used in the qualitative and quantitative comparions
                    # val / test: use the first frame as the source mesh, use the randomly sample frame as the target mesh
                    for frame_name in frame_names:
                        if int(frame_name)>0:
                            idx_cano                = idx_motion
                            cano_seq_name           = motion_seq_name
                            cano_seq_frame_name     = "0000"
                            if "cat" in motion_seq_name or "lion" in motion_seq_name:
                                motion_seq_frame_name0  = "0003"
                            elif "horse" in motion_seq_name:
                                motion_seq_frame_name0  = "0005"
                            else:
                                motion_seq_frame_name0  = "0001"
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
                        idx_cano                = idx_motion
                        cano_seq_name           = motion_seq_name
                        cano_seq_frame_name     = "0000"
                        motion_seq_frame_name   = frame_name
                        self.all_deform_pairs.append({
                                "pair_info": (idx_cano, cano_seq_name, cano_seq_frame_name, 
                                    idx_motion, cano_seq_name, cano_seq_frame_name, motion_seq_name, motion_seq_frame_name)
                                })

        if self.motion_split[:5] == "train" or self.num_sampled_pairs > 0:
            self.random_shuffle_samples(self.num_sampled_pairs)
        else:
            self.sample_deform_pairs = self.all_deform_pairs
    
