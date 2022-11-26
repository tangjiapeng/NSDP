import os
import numpy as np
import open3d as o3d
from utils.visualize import vis_error_map
import trimesh

def create_directory(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

def create_directories_and_files(output_dir, meta_data, ext):
    idx_cano, cano_seq_name, cano_seq_frame_name, \
            idx_motion, src_seq_name, src_seq_frame_name, tgt_seq_name, tgt_seq_frame_name = meta_data["pair_info"]

    src_dir = os.path.join(output_dir,  "source")
    create_directory(src_dir)
    src_file = os.path.join(src_dir, "{}_{}.{}".format(src_seq_name, src_seq_frame_name, ext))

    cano_dir = os.path.join(output_dir,  "canonical")
    create_directory(cano_dir)
    cano_file = os.path.join(cano_dir, "{}_{}.{}".format(cano_seq_name, cano_seq_frame_name, ext))

    deform_dir = os.path.join(output_dir, "deformed")
    create_directory(deform_dir)
    deform_file = os.path.join(deform_dir, "{}_{}_to_{}_{}.{}".format(src_seq_name, src_seq_frame_name, tgt_seq_name, tgt_seq_frame_name, ext))

    target_dir = os.path.join(output_dir,  "target")
    create_directory(target_dir)
    target_file = os.path.join(target_dir, "{}_{}_to_{}_{}.{}".format(src_seq_name, src_seq_frame_name, tgt_seq_name, tgt_seq_frame_name, ext))

    handle_dir = os.path.join(output_dir,  "handle")
    create_directory(handle_dir)
    handle_file = os.path.join(handle_dir, "{}_{}_to_{}_{}.{}".format(src_seq_name, src_seq_frame_name, tgt_seq_name, tgt_seq_frame_name, ext))

    return src_file, cano_file, deform_file, target_file, handle_file

def generate_meshes(output_dir, out_dict, meta_data, ext, vert_pred_color=False):
    src_file, cano_file, deform_file, target_file, handle_file = create_directories_and_files(output_dir, meta_data, ext)

    verts_pred  = out_dict["verts_tgt_pred"].squeeze().cpu().numpy()
    verts_cano  = out_dict["verts_cano"].squeeze().cpu().numpy()
    verts_src   = out_dict["verts_src"].squeeze().cpu().numpy()
    verts_tgt   = out_dict["verts_tgt"].squeeze().cpu().numpy()
    cano_handle_vert_idx = out_dict["cano_handle_vert_idx"].squeeze().cpu().numpy()
    faces       = out_dict["faces"].squeeze().cpu().numpy()
    
    # export source mesh
    verts_src_colors = np.ones(verts_src.shape, dtype=np.float32) * 0.75
    src_handle_colors = np.array([[1.0, 0.0, 0.0]], dtype=np.float32)
    verts_src_colors[cano_handle_vert_idx, :] = src_handle_colors.repeat(cano_handle_vert_idx.sum(), axis=0)
    mesh_src = trimesh.Trimesh(verts_src, faces, vertex_colors=(verts_src_colors*255).astype('uint8'), process=False)
    mesh_src.export(src_file)
    
    # export canonical mesh
    verts_cano_colors = verts_src_colors
    mesh_cano = trimesh.Trimesh(verts_cano, faces, vertex_colors=(verts_cano_colors*255).astype('uint8'), process=False)
    mesh_cano.export(cano_file)
    
    # export deformed/predicted mesh
    if vert_pred_color:
        mesh_deform = vis_error_map(verts_pred, faces, np.sqrt(((verts_pred- verts_tgt)**2).sum(-1)))
        o3d.io.write_triangle_mesh(deform_file, mesh_deform)
    else:
        mesh_deform = trimesh.Trimesh(verts_pred, faces, process=False)
        mesh_deform.export(deform_file)

    # export target mesh
    verts_tgt_colors = np.ones(verts_tgt.shape, dtype=np.float32) * 0.75
    tgt_handle_colors = np.array([[0.0, 0.0, 1.0]], dtype=np.float32)
    verts_tgt_colors[cano_handle_vert_idx, :] = tgt_handle_colors.repeat(cano_handle_vert_idx.sum(), axis=0)
    mesh_tgt = trimesh.Trimesh(verts_tgt, faces, vertex_colors=(verts_tgt_colors*255).astype('uint8'), process=False)
    mesh_tgt.export(target_file)
    
    # export handle mesh
    face_reshape = faces.reshape(-1)
    face_select_mask = cano_handle_vert_idx[face_reshape].reshape(-1, 3)
    face_select_mask = (face_select_mask.sum(axis=1)) == 3
    mesh_handle = mesh_tgt.copy()
    mesh_handle.update_faces(face_select_mask)
    mesh_handle.export(handle_file)


def generate_pointclouds(output_dir, out_dict, meta_data, ext):
    src_file, cano_file, deform_file, target_file, handle_file = create_directories_and_files(output_dir, meta_data, ext)
    

    # we use the first 3 channels of surface_samples_inputs to obtain pc_src, 
    # as we maybe have done some transformation on source point cloud 
    pc_src      = out_dict['surface_samples_inputs'].squeeze().cpu().numpy()[:, 0:3]
    pc_handle   = out_dict['surface_samples_inputs'].squeeze().cpu().numpy()[:, 3:6]
    handle_mask = out_dict['surface_samples_inputs'].squeeze().cpu().numpy()[:, 6]
    pc_deform   = out_dict["surface_samples_tgt_pred"].squeeze().cpu().numpy()
    
    pc_tgt      = out_dict["surface_samples_tgt"].squeeze().cpu().numpy()
    pc_cano     = out_dict["surface_samples_cano"].squeeze().cpu().numpy()
    
    # export source point cloud
    pc_src_o3d = o3d.geometry.PointCloud(points=o3d.utility.Vector3dVector(pc_src))
    pc_src_colors = np.ones(pc_src.shape, dtype='float32') * 0.75
    pc_src_colors[handle_mask>0, :] = np.array([1.0, 0.0, 0.0], dtype='float32')[None, :].repeat(handle_mask.sum(),axis=0)
    pc_src_o3d.colors = o3d.utility.Vector3dVector(pc_src_colors)
    o3d.io.write_point_cloud(src_file, pc_src_o3d)
    
    # export canonical point cloud
    pc_cano_o3d = o3d.geometry.PointCloud(points=o3d.utility.Vector3dVector(pc_cano))
    pc_cano_colors = np.ones(pc_cano.shape, dtype='float32') * 0.75 
    pc_cano_colors[handle_mask>0, :] = np.array([1.0, 0.0, 0.0], dtype='float32')[None, :].repeat(handle_mask.sum(),axis=0)
    pc_cano_o3d .colors = o3d.utility.Vector3dVector(pc_cano_colors)
    o3d.io.write_point_cloud(cano_file, pc_cano_o3d)
    
    # export deformed/predicted point cloud
    pc_deform_o3d = o3d.geometry.PointCloud(points=o3d.utility.Vector3dVector(pc_deform))
    o3d.io.write_point_cloud(deform_file, pc_deform_o3d)
    
    # export target point cloud
    pc_tgt_o3d = o3d.geometry.PointCloud(points=o3d.utility.Vector3dVector(pc_tgt))
    pc_tgt_colors = np.ones(pc_tgt.shape, dtype='float32') * 0.75
    pc_tgt_colors[handle_mask>0, :] = np.array([0.0, 0.0, 1.0], dtype='float32')[None, :].repeat(handle_mask.sum(),axis=0)
    pc_tgt_o3d.colors = o3d.utility.Vector3dVector(pc_tgt_colors)
    o3d.io.write_point_cloud(target_file, pc_tgt_o3d)
    
    # export handle point cloud
    pc_handles_o3d = o3d.geometry.PointCloud(points=o3d.utility.Vector3dVector(pc_handle))
    pc_handles_colors = np.array([0.0, 0.0, 1.0], dtype='float32')[None, :].repeat(handle_mask.sum(),axis=0)
    pc_handles_o3d.colors = o3d.utility.Vector3dVector(pc_handles_colors)
    o3d.io.write_point_cloud(handle_file, pc_handles_o3d)


def define_userhandle_folder_name( cfg ):
    partial_range  = cfg['data']['partial_range'] 
    
    dirname = "drag"
    
    if cfg['data']['userhandle']['head']:
        dirname += "_head"

    elif cfg['data']['userhandle']['tail']:
        dirname += "_tail"

    elif cfg['data']['userhandle']['frontleftfoot']:
        dirname += "_frontleftfoot"

    elif cfg['data']['userhandle']['frontrightfoot']:
        dirname += "_frontrightfoot"

    elif cfg['data']['userhandle']['behindleftfoot']:
        dirname += "_behindleftfoot"

    elif cfg['data']['userhandle']['behindrightfoot']:
        dirname += "_behindrightfoot"
        
    dx, dy, dz = cfg['data']['userhandle']['xtrans'], cfg['data']['userhandle']['ytrans'], cfg['data']['userhandle']['ztrans']

    dirname += "_x%.2fy%.2fz%.2f"%(dx, dy, dz)
            
    dirname += "_ratio%.2f"%partial_range

    if cfg['data']['userhandle']['cliptail']:
        dirname += "_cliptail"
    
    return dirname
    