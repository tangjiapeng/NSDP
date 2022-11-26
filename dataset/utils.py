import sys,os
import numpy as np
import trimesh
from scipy.spatial import KDTree

##################################################################################################################
# the below function are used to load files and fix coord system if the coord system is not consistent
def load_npz_surface_flow(path):
    flow_dict = np.load(path)
    points = flow_dict['points'].astype(np.float32)
    normals = flow_dict['normals'].astype(np.float32)
    return points, normals

def load_npz_space_flow(path):
    flow_dict = np.load(path)
    points = flow_dict['points'].astype(np.float32)
    return points

def load_mesh_info(path):
    mesh = trimesh.load_mesh(path, process=False)
    verts = np.array(mesh.vertices).astype(np.float32)
    edges = np.array(mesh.edges).astype(np.int64)
    reverse_edges = np.stack([edges[:, 1], edges[:, 0]], axis=-1)
    edges = np.concatenate([edges, reverse_edges], axis=0)
    faces = np.array(mesh.faces).astype(np.int64)
    return verts, edges, faces

def fix_coord_system(points):
    #from x, y, z to x, z, -y for the deformation transfer dataset
    points_xzy = np.stack([points[:, 0], -points[:, 2], points[:, 1]], axis=1)
    return np.ascontiguousarray(points_xzy)
##################################################################################################################


##################################################################################################################
# the below function are used to add some transformations on data, like subsampling, adding noises to source mesh,
# creating incomplete parts, computing the handle mask, normalization mesh, etc.
def subsample_surface_flow(cfg, surface_samples_cano, surface_samples_src, surface_samples_tgt, surface_samples_idxs=None): 
    num_surf_samples = cfg['data']['num_surf_samples']
    if surface_samples_idxs is None: 
        surface_samples_idxs = np.random.permutation(surface_samples_cano.shape[0])[:num_surf_samples]  
    surface_samples_cano = surface_samples_cano[surface_samples_idxs, :]
    surface_samples_src  = surface_samples_src[surface_samples_idxs, :]
    surface_samples_tgt  = surface_samples_tgt[surface_samples_idxs, :]
    return surface_samples_cano, surface_samples_src, surface_samples_tgt, surface_samples_idxs

def subsample_space_flow(cfg, flow_samples_cano, flow_samples_src, flow_samples_tgt):
    num_space_samples = cfg['data']['num_space_samples']
    if flow_samples_cano.shape[0] > num_space_samples:
        flow_samples_idxs = np.random.permutation(flow_samples_cano.shape[0])[:num_space_samples]
        flow_samples_cano = flow_samples_cano[flow_samples_idxs, :]
        flow_samples_src  = flow_samples_src[flow_samples_idxs, :]
        flow_samples_tgt  = flow_samples_tgt[flow_samples_idxs, :]  
    return flow_samples_cano, flow_samples_src, flow_samples_tgt

def cano_sample_handle_mask(cfg, surface_samples_cano, bbox_min, bbox_max):
    partial_range   = cfg['data']['partial_range'] 
    head_sample_idx = surface_samples_cano[:, 1] < bbox_min[1] + partial_range
    tail_sample_idx = surface_samples_cano[:, 1] > bbox_max[1] - partial_range
    foot_sample_idx = surface_samples_cano[:, 2] < bbox_min[2] + partial_range
    handle_sample_idx = head_sample_idx | tail_sample_idx | foot_sample_idx
    return handle_sample_idx

def cano_vert_handle_mask(cfg, vert_cano_norm, vert_bbox_min, vert_bbox_max):
    partial_range   = cfg['data']['partial_range'] 
    head_vert_idx = vert_cano_norm[:, 1] < vert_bbox_min[1] + partial_range
    tail_vert_idx = (vert_cano_norm[:, 1] > vert_bbox_max[1] - partial_range ) #& (vert_cano_norm[:, 2] > -partial_range)
    foot_vert_idx = vert_cano_norm[:, 2] < vert_bbox_min[2] + partial_range  
    handle_vert_idx = (head_vert_idx + tail_vert_idx + foot_vert_idx) > 0 
    return handle_vert_idx

def add_noise_to_src(cfg, surface_samples_src):
    noise_level     = cfg['data']['noise_level']
    noise = noise_level * np.random.randn(*surface_samples_src.shape).astype(np.float32)
    surface_samples_src_noise = surface_samples_src + noise
    #surface_samples_tgt_noise = surface_samples_tgt + noise
    return surface_samples_src_noise

def create_partial_src(cfg, surface_samples_src, handle_sample_idx, num_seeds = 5):
    partial_shape_ratio = cfg['data']['partial_shape_ratio']
    # create incompleteness on non-handles regions
    if partial_shape_ratio < 1.0:
        nonhandle_sample_idx = ~handle_sample_idx
        hole_ratio = 1.0 - partial_shape_ratio
        # num of seeds for holes
        num_samples_remained = int(partial_shape_ratio * len(surface_samples_src))
        num_samples_per_hole = int(hole_ratio * len(surface_samples_src) // num_seeds)
        # select hole seeds on non-handles regions
        surface_samples_src_nonhandle = surface_samples_src[nonhandle_sample_idx]
        select_seeds_idx = np.random.permutation(surface_samples_src_nonhandle.shape[0])[:num_seeds]
        surface_samples_src_nonhandle_holeseeds = surface_samples_src_nonhandle[select_seeds_idx]
        # search num_sample_per_hole points on surface samples
        kdtree = KDTree(surface_samples_src)
        _, remove_idx = kdtree.query(surface_samples_src_nonhandle_holeseeds, k=num_samples_per_hole)
        remove_idx = remove_idx.reshape(-1)
        # get the remain_idx
        remain_idx = set(np.arange(len(surface_samples_src))) - set(remove_idx)
        remain_idx = np.array(list(remain_idx))
    else:
        remain_idx = np.arange(len(surface_samples_src))
    return remain_idx

def normalize_origin_mesh(vertices, orig2world):
    vertices_normalize = (np.matmul(orig2world[:3, :3], vertices.T) + orig2world[:3, 3:4]).T
    return vertices_normalize
##################################################################################################################


def cano_handle_user_define(cfg, vert_cano_norm, vert_bbox_min, vert_bbox_max, vert_src_norm):
    # define user specified handle mask
    partial_range   = cfg['data']['partial_range'] 
    head_vert_idx = vert_cano_norm[:, 1] < vert_bbox_min[1] + partial_range
    if cfg['data']['userhandle']['cliptail']:
        tail_vert_idx = (vert_cano_norm[:, 1] > vert_bbox_max[1] - partial_range ) & (vert_cano_norm[:, 2] > -partial_range)
    else:
        tail_vert_idx = vert_cano_norm[:, 1] > vert_bbox_max[1] - partial_range
    foot_vert_idx = vert_cano_norm[:, 2] < vert_bbox_min[2] + partial_range  
    handle_vert_idx = (head_vert_idx + tail_vert_idx + foot_vert_idx) > 0 
    
    # define the handle mask for four feet
    leftfoot_vert_idx =  foot_vert_idx & (vert_cano_norm[:, 0] > 0) 
    rightfoot_vert_idx = foot_vert_idx & (vert_cano_norm[:, 0] < 0)
    frontfoot_vert_idx =  foot_vert_idx & (vert_cano_norm[:, 1] < 0)
    behindfoot_vert_idx = foot_vert_idx & (vert_cano_norm[:, 1] > 0) 
    frontleftfoot_vert_idx = leftfoot_vert_idx & frontfoot_vert_idx
    frontrightfoot_vert_idx = rightfoot_vert_idx & frontfoot_vert_idx
    behindleftfoot_vert_idx = leftfoot_vert_idx & behindfoot_vert_idx
    behindrightfoot_vert_idx = rightfoot_vert_idx & behindfoot_vert_idx
    
    if cfg['data']['userhandle']['head']:
        move_vert_idx = head_vert_idx
    elif cfg['data']['userhandle']['tail']:
        move_vert_idx = tail_vert_idx
    elif cfg['data']['userhandle']['frontleftfoot']:
        move_vert_idx = frontleftfoot_vert_idx
    elif cfg['data']['userhandle']['frontrightfoot']:
        move_vert_idx = frontrightfoot_vert_idx
    elif cfg['data']['userhandle']['behindleftfoot']:
        move_vert_idx = behindleftfoot_vert_idx
    elif cfg['data']['userhandle']['behindrightfoot']:
        move_vert_idx = behindrightfoot_vert_idx
        
    dx, dy, dz = cfg['data']['userhandle']['xtrans'], cfg['data']['userhandle']['ytrans'], cfg['data']['userhandle']['ztrans']
    handle_displace = np.array([[dx, dy, dz]], dtype='f4').repeat(vert_src_norm.shape[0], axis=0) * move_vert_idx[:, None]
    #print(handle_displace.shape, vert_src_norm.shape, move_vert_idx.shape)
    vert_tgt_norm = vert_src_norm + handle_displace
    return handle_vert_idx, vert_tgt_norm
    
