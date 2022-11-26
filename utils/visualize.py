import os, sys
import numpy as np
import torch
import open3d as o3d
import trimesh

def get_jet_color(v, vmin=0.0, vmax=1.0):
    """
    Maps
        map a vector clipped with the range [vmin, vmax] to colors
    Args:
        - vec (): 
    """ 
    c = np.array([1.0, 1.0, 1.0], dtype='float32')
    
    if v < vmin:
        v = vmin
    if v > vmax:
        v = vmax
    dv = vmax - vmin

    if v < (vmin + 0.25 * dv):
        c[0] = 0
        c[1] = 4 * (v - vmin) / dv
    elif v < vmin + 0.5 * dv:
        c[0] = 0
        c[2] = 1 + 4 * (vmin + 0.25 * dv - v) / dv
    elif v < vmin + 0.75 * dv:
        c[0] = 4 * (v - vmin - 0.5 * dv) / dv
        c[2] = 0
    else:
        c[1] = 1 + 4 * (vmin + 0.75 * dv - v) / dv
        c[2] = 0
    return c

def vis_error_map(vertices, faces, error_npy, use_max=False):
    """
    Maps
        Convert a mesh with vertices errors to a colored mesh
        using Jet_color_map
    Args:
        - vertices: Nx3
        - faces:    Fx3
        - error_npy: N
    """ 
    error_min  = 0
    if use_max:
        error_max = error_npy.max()
    else:
        error_max  = 0.2 
    error_dist = error_max - error_min
    num_points  = error_npy.shape[0]
    error_map  = np.ones((num_points, 3), dtype='float32')

    mask = error_npy < error_min + 0.25 * error_dist
    error_map[mask, 0] = np.zeros((mask.sum()), dtype='float32')
    error_map[mask, 1] = 4 * (error_npy[mask] - error_min) / error_dist
    
    mask = (error_npy >= error_min + 0.25 * error_dist) & (error_npy < error_min + 0.5 * error_dist)
    error_map[mask, 0] = np.zeros((mask.sum()), dtype='float32')
    error_map[mask, 2] = 1 + 4 * (error_min + 0.25 * error_dist - error_npy[mask]) / error_dist
    
    mask = (error_npy >= error_min + 0.5 * error_dist) & (error_npy < error_min + 0.75 * error_dist)
    error_map[mask, 0] = 4 * (error_npy[mask] - error_min - 0.5 * error_dist) / error_dist
    error_map[mask, 2] = np.zeros((mask.sum()), dtype='float32')
    
    mask = error_npy >= error_min + 0.75 * error_dist
    error_map[mask, 1] = 1 + 4 * (error_min + 0.75 * error_dist - error_npy[mask]) / error_dist
    error_map[mask, 2] = np.zeros((mask.sum()), dtype='float32')
    
    # trimesh
    # mesh = trimesh.Trimesh(vertices=vertices, faces=faces, vertex_colors=error_map, process=False)
    # mesh.visual.vertex_colors = error_map
    
    # open3d
    mesh = o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(vertices), o3d.utility.Vector3iVector(faces))
    mesh.vertex_colors = o3d.utility.Vector3dVector(error_map)
    
    return mesh


def vector_magnitude(vec):
    """
    Calculates a vector's magnitude.
    Args:
        - vec (): 
    """
    magnitude = np.sqrt(np.sum(vec**2))
    return(magnitude)


def calculate_zy_rotation_for_arrow(vec):
    """
    Calculates the rotations required to go from the vector vec to the 
    z axis vector of the original FOR. The first rotation that is 
    calculated is over the z axis. This will leave the vector vec on the
    XZ plane. Then, the rotation over the y axis. 

    Returns the angles of rotation over axis z and y required to
    get the vector vec into the same orientation as axis z
    of the original FOR

    Args:
        - vec (): 
    """
    # Rotation over z axis of the FOR
    gamma = np.arctan(vec[1]/vec[0])
    Rz = np.array([[np.cos(gamma),-np.sin(gamma),0],
                [np.sin(gamma),np.cos(gamma),0],
                [0,0,1]])
    # Rotate vec to calculate next rotation
    vec = Rz.T@vec.reshape(-1,1)
    vec = vec.reshape(-1)
    # Rotation over y axis of the FOR
    beta = np.arctan(vec[0]/vec[2])
    Ry = np.array([[np.cos(beta),0,np.sin(beta)],
                [0,1,0],
                [-np.sin(beta),0,np.cos(beta)]])
    return(Rz, Ry)

def create_arrow(scale=10):
    """
    Create an arrow in for Open3D
    """
    cone_height = scale*0.2
    cylinder_height = scale*0.8
    cone_radius = scale/10
    cylinder_radius = scale/20
    mesh_frame = o3d.geometry.TriangleMesh.create_arrow(
        cone_radius=cone_radius,
        cylinder_radius=cylinder_radius,
        cone_height=cone_height, cylinder_height=cylinder_height)
    #cylinder_radius=0.00175, cone_radius=0.0035,  cylinder_height=0.02, cone_height=0.01, resolution=10)
    # mesh_arrow = o3d.geometry.TriangleMesh.create_arrow(cylinder_radius=0.00175, cone_radius=0.0035, cylinder_height=0.02, cone_height=0.01, resolution=10)
    return(mesh_frame)

def get_arrow(origin=[0, 0, 0], end=None, vec=None):
    """
    Creates an arrow from an origin point to an end point,
    or create an arrow from a vector vec starting from origin.
    Args:
        - end (): End point. [x,y,z]
        - vec (): Vector. [i,j,k]
    """
    scale = 10
    Ry = Rz = np.eye(3)
    T = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
    T[:3, -1] = origin
    if end is not None:
        vec = np.array(end) - np.array(origin)
    elif vec is not None:
        vec = np.array(vec)
    if end is not None or vec is not None:
        scale = vector_magnitude(vec)
        Rz, Ry = calculate_zy_rotation_for_arrow(vec)
    mesh = create_arrow(scale)
    # Create the arrow
    mesh.rotate(Ry, center=np.array([0, 0, 0]))
    mesh.rotate(Rz, center=np.array([0, 0, 0]))
    mesh.translate(origin)
    return(mesh)

def merge_meshes(meshes):
    # Compute total number of vertices and faces.
    num_vertices = 0
    num_triangles = 0
    num_vertex_colors = 0
    for i in range(len(meshes)):
        num_vertices += np.asarray(meshes[i].vertices).shape[0]
        num_triangles += np.asarray(meshes[i].triangles).shape[0]
        num_vertex_colors += np.asarray(meshes[i].vertex_colors).shape[0]

    # Merge vertices and faces.
    vertices = np.zeros((num_vertices, 3), dtype=np.float64)
    triangles = np.zeros((num_triangles, 3), dtype=np.int32)
    vertex_colors = np.zeros((num_vertex_colors, 3), dtype=np.float64)

    vertex_offset = 0
    triangle_offset = 0
    vertex_color_offset = 0
    for i in range(len(meshes)):
        current_vertices = np.asarray(meshes[i].vertices)
        current_triangles = np.asarray(meshes[i].triangles)
        current_vertex_colors = np.asarray(meshes[i].vertex_colors)

        vertices[vertex_offset:vertex_offset + current_vertices.shape[0]] = current_vertices
        triangles[triangle_offset:triangle_offset + current_triangles.shape[0]] = current_triangles + vertex_offset
        vertex_colors[vertex_color_offset:vertex_color_offset + current_vertex_colors.shape[0]] = current_vertex_colors

        vertex_offset += current_vertices.shape[0]
        triangle_offset += current_triangles.shape[0]
        vertex_color_offset += current_vertex_colors.shape[0]

    # Create a merged mesh object.
    mesh = o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(vertices), o3d.utility.Vector3iVector(triangles))
    mesh.paint_uniform_color([1, 0, 0])
    mesh.vertex_colors = o3d.utility.Vector3dVector(vertex_colors)
    return mesh


def vis_flow_volume_arrow(flow_volume, flow_mask, dim = 32, bbox_size = 1.5):            
    """
    Visualize a low-resolution volumetric flow using arrow meshes
    the direction of arrow indicates the flow direction
    the length of arrow represents the magnititude of flow vector
    Args:
        - flow_volume: Nx3  reshaped from [H, W, D, 3]
        - flow_mask:   N    reshaped from [H, W, D]  indicates the valid flow region as we only flow in the near surface regions
        - dim: the resolution of volumetric flow grid
        - bbox: the actual bounding box size
    """
    flow_lenght = np.sqrt(np.sum(flow_volume**2, axis=1))
    min_len, max_len = flow_lenght[flow_mask==1].min(), flow_lenght[flow_mask==1].max()
    
    arrow_triangles = []
    for idx in range(flow_volume.shape[0]):
        magnitude = vector_magnitude(flow_volume[idx, :]+1e-6)
        Rz, Ry = calculate_zy_rotation_for_arrow(flow_volume[idx, :]+1e-6)
        if flow_mask[idx]:
        
            # Create the index coordinates (int) to a  grid point (float)
            z, y, x = idx // (dim * dim), (idx // dim) % dim, idx % dim
            z_coord, y_coord, x_coord = ((z + 0.5)/dim-0.5)*bbox_size, ((y+ 0.5)/dim-0.5)*bbox_size, ((x+ 0.5)/dim-0.5)*bbox_size
            center = np.array([x_coord, y_coord, z_coord], dtype='f4')
            scale = np.array([1, 1, 1], dtype='f4')

            # Create a sphere mesh.
            if True:
                mesh_arrow = o3d.geometry.TriangleMesh.create_arrow(cylinder_radius=0.007, cone_radius=0.014,  cylinder_height=0.08, cone_height=0.04, resolution=10)
                scale = scale + 1e-8

                T_t = np.eye(4)
                T_t[0:3, 3] = center

                T_s = np.eye(4)
                T_s[0, 0] = scale[0]
                T_s[1, 1] = scale[1]
                T_s[2, 2] = scale[2]

                T_R = np.eye(4)
                T = np.matmul(T_t, np.matmul(T_R, T_s))
                
                mesh_arrow.transform(T_s)
                mesh_arrow.rotate(Ry, center=np.array([0, 0, 0]))
                mesh_arrow.rotate(Rz, center=np.array([0, 0, 0]))
                mesh_arrow.transform(T_t)
            else:
                mesh_arrow = get_arrow(origin=center, vec=flow_volume[idx, :])

            # We view spheres as wireframe.
            mesh_arrow.paint_uniform_color(get_jet_color(flow_lenght[idx], min_len, max_len))
            arrow_triangles.append(mesh_arrow)

    # Merge sphere meshes.
    merged_arrow_triangles = merge_meshes(arrow_triangles)

    return merged_arrow_triangles

def vis_flow_surface_arrow(geometry, flow, mask):
    """
    Visualize a point cloud with flow using arrow meshes
    the direction of arrow indicates the flow direction
    the length of arrow represents the magnititude of flow vector
    Args:
        - geometry: Nx3   represents the point cloud
        - flow:     Nx3   represents the flow vectors of points
        - mask:     Nx3   indicates the valid flow region, e.g. the handle regions
    """
    flow_lenght = np.sqrt(np.sum(flow**2, axis=1))
    min_len, max_len = flow_lenght[mask==1].min(), flow_lenght[mask==1].max()
    
    arrow_triangles = []
    for idx in range(flow.shape[0]):
        magnitude = vector_magnitude(flow[idx, :]+1e-6)
        Rz, Ry = calculate_zy_rotation_for_arrow(flow[idx, :]+1e-6)

        center = geometry[idx, :]
        scale = np.array([1, 1, 1], dtype='f4')

        if mask[idx]: 

            # Create a sphere mesh.
            if True:
                mesh_arrow = o3d.geometry.TriangleMesh.create_arrow(cylinder_radius=0.007, cone_radius=0.014,  cylinder_height=0.08, cone_height=0.04, resolution=10)

                scale = scale + 1e-8

                T_t = np.eye(4)
                T_t[0:3, 3] = center

                T_s = np.eye(4)
                T_s[0, 0] = scale[0]
                T_s[1, 1] = scale[1]
                T_s[2, 2] = scale[2]

                T_R = np.eye(4)
                T = np.matmul(T_t, np.matmul(T_R, T_s))
                
                mesh_arrow.transform(T_s)
                mesh_arrow.rotate(Ry, center=np.array([0, 0, 0]))
                mesh_arrow.rotate(Rz, center=np.array([0, 0, 0]))
                mesh_arrow.transform(T_t)
            else:
                mesh_arrow = get_arrow(origin=center, vec=flow[idx, :])

            # We view spheres as wireframe.
            mesh_arrow.paint_uniform_color(get_jet_color(flow_lenght[idx], min_len, max_len))
            arrow_triangles.append(mesh_arrow)

    # Merge sphere meshes.
    merged_arrow_triangles = merge_meshes(arrow_triangles)
    return merged_arrow_triangles