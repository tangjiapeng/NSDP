import os
import numpy as np
from scipy.spatial import KDTree
import trimesh

def compute_dist_square(vertices, vertices_gt):
    l2 = ((vertices - vertices_gt)**2).sum(-1)
    return l2.mean()


def normal_consistency(normals_src, normals_tgt):
    normals_src = normals_src / np.linalg.norm(normals_src, axis=-1, keepdims=True)
    normals_tgt = normals_tgt / np.linalg.norm(normals_tgt, axis=-1, keepdims=True)

    normals_dot_product = (normals_tgt * normals_src).sum(axis=-1)
    # Handle normals that point into wrong direction gracefully
    # (mostly due to mehtod not caring about this in generation)
    normals_dot_product = np.abs(normals_dot_product)
    
    return normals_dot_product.mean()

def chamfer_distance(points, points_gt):
    kdtree = KDTree(points_gt)
    compleness, idx = kdtree.query(points)
    kdtree = KDTree(points)
    accuracy, idx = kdtree.query(points_gt)
    chamfer_l1 = 0.5 * (accuracy.mean() + compleness.mean())
    
    return chamfer_l1


def compute_evaluation_metrics(out_dict):
    eval_dict = {}
    # convert to trimesh
    verts_pred = out_dict["verts_tgt_pred"].squeeze().cpu().detach().numpy()
    verts_gt   = out_dict["verts_tgt"].squeeze().cpu().numpy()
    faces      = out_dict["faces"].squeeze().cpu().numpy()
    mesh_pred   = trimesh.Trimesh(verts_pred, faces, process=False)
    mesh_gt    = trimesh.Trimesh(verts_gt, faces, process=False)

    # l2 
    l2_error = compute_dist_square(verts_pred, verts_gt)
    eval_dict['l2'] = l2_error
    
    # normal consistency
    face_normals_pred = mesh_pred.face_normals.astype(np.float32)
    face_normals_gt   = mesh_gt.face_normals.astype(np.float32)
    fnc = normal_consistency(face_normals_pred, face_normals_gt)
    eval_dict['fnc']  = fnc

    # CD
    pointcloud_size = 30000
    _, face_idx = mesh_pred.sample(pointcloud_size, return_index=True)
    alpha = np.random.dirichlet((1,)*3, pointcloud_size)
    v_pred = verts_pred[faces[face_idx]]
    points_pred = (alpha[:, :, None] * v_pred).sum(axis=1)
    v_gt   = verts_gt[faces[face_idx]]
    points_gt   = (alpha[:, :, None] * v_gt).sum(axis=1)
    chamfer_l1 = chamfer_distance(points_pred, points_gt)
    eval_dict['cd']  = chamfer_l1

    return eval_dict