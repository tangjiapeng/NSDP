from concurrent.futures import process
import os
from posixpath import basename
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import tqdm
from joblib import Parallel, delayed
from absl import flags
from absl import app
import glob
import argparse
import open3d as o3d
import subprocess as sp
import numpy as np
import trimesh


def process_one(temp_sample_info, mesh_paths, mesh_directory, dataset_directory, skip_existing):
    """Processes a single mesh, adding it to the dataset."""
    name = os.path.basename(mesh_paths[-1])
    name, extension = os.path.splitext(name)
    valid_extensions = ['.obj', '.ply', '.off'] 
    if extension not in valid_extensions:
        raise ValueError(f'File with unsupported extension {extension} found: {f}.'
                         f' Only {valid_extensions} are supported.')
    # define the generate filename using sequence_idx
    idx_of_name = int(name.split('-')[-1])
    name = '%04d'%idx_of_name
    output_dir = f'{dataset_directory}/{name}/'
    
    # This is the last file the processing writes, if it already exists the
    # example has already been processed.
    if not skip_existing or not os.path.isfile(f'{output_dir}/flow.npz'):
        export_pointcloud(temp_sample_info, mesh_paths, mesh_directory, dataset_directory)
    else:
        print(f'Skipping shell script processing for {output_dir},'
              ' the output already exists.')

    return output_dir

def export_pointcloud(temp_sample_info, mesh_paths, mesh_directory, dataset_directory):
    
    nf = len(mesh_paths)
    face_idx = temp_sample_info['face_idx']
    alpha = temp_sample_info['alpha']
    noise = temp_sample_info['noise']
    
    for i in range(nf):
        input_mesh_file = mesh_paths[i]
        mesh = trimesh.load_mesh(input_mesh_file, process=False)
        name = os.path.basename(mesh_paths[i])
        name, extension = os.path.splitext(name)
        # define the generate filename using sequence_idx
        idx_of_name = int(name.split('-')[-1])
        name = '%04d'%idx_of_name
        output_dir = f'{dataset_directory}/{name}/'
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        nearsurf_pointcloud_file = os.path.join(output_dir, 'flow.npz')
        
        norm_param_filefolder = output_dir
        norm_param_file = os.path.join(norm_param_filefolder , 'orig_to_gaps.txt')
        orig2world = np.reshape(np.loadtxt(norm_param_file), [4, 4]) 
        
        # normalization 
        R, t = orig2world[:3, :3], orig2world[:3, 3:4]
        scale, loc = R[0, 0], t.squeeze(1)
        mesh.apply_scale(scale)
        mesh.apply_translation(loc)
        
        # sample near surface points
        vertices = mesh.vertices
        faces = mesh.faces
        v = vertices[faces[face_idx]]
        normals = mesh.face_normals[face_idx]
        points = (alpha[:, :, None] * v).sum(axis=1)
        points_nearsurf = points + normals * noise
        
        # Compress
        float16 = True
        if float16:
            dtype = np.float16
        else:
            dtype = np.float32
        points_nearsurf = points_nearsurf.astype(dtype)
        loc = loc.astype(dtype)
        scale = scale.astype(scale)
        
        print('Writing near surface pointcloud: %s' % nearsurf_pointcloud_file, points_nearsurf.shape)
        np.savez(nearsurf_pointcloud_file, points=points_nearsurf, loc=loc, scale=scale)
        

def generate_flow(mesh_directory, dataset_directory, mesh_format, temp_lst, skip_existing=True, max_threads=-1, pointcloud_size=200000, sigma1=0.1, sigma2=0.02):
    # Make the directories first because it's not threadsafe and also might fail.
    print('Creating directories...')
    if not os.path.isdir(f'{dataset_directory}'):
        os.makedirs(f'{dataset_directory}')
        
    print("Get Sampling Info of animals temaplates")
    template_modnames = open(temp_lst, 'r').readlines()
    template_modnames = [f.strip() for f in template_modnames]
    
    sample_info = {}
    for f in template_modnames:
        if f == '':
            continue
        template_name = f #.split('_')[0]
        try:
            template_mesh_path = os.path.join(mesh_directory, f,  f+"-00.obj")
            mesh = trimesh.load_mesh(template_mesh_path, process=False)
        except:
            template_mesh_path = os.path.join(mesh_directory, f,  f.split('-')[0]+"-00.obj")
            mesh = trimesh.load_mesh(template_mesh_path, process=False)
        
        _, face_idx = mesh.sample(pointcloud_size, return_index=True)
        alpha = np.random.dirichlet((1,)*3, pointcloud_size)
        noise1 = 2.0 * np.random.rand(int(pointcloud_size//2), 1) - 1.0 
        noise1 = noise1 * sigma1
        noise2 = 2.0 * np.random.rand(int(pointcloud_size//2), 1) - 1.0 
        noise2 = noise2 * sigma2
        noise = np.concatenate([noise1, noise2], axis=0)
        sample_info[template_name] = { 'face_idx': face_idx, 'alpha' : alpha, 'noise' : noise }
    
    model_files = []
    for f in sorted(os.listdir(mesh_directory)):
        if os.path.isdir(os.path.join(mesh_directory, f)): 
            template_name = f #.split('_')[0]
            if template_name not in sample_info.keys():
                print(f, "is not in the selected templates")
                continue
            mesh_modelname_directory = os.path.join(mesh_directory, f)
            dataset_modelname_directory = os.path.join(dataset_directory, f)
            print('Creating model directories...')
            if not os.path.isdir(f'{dataset_modelname_directory}'):
                os.makedirs(f'{dataset_modelname_directory}')
    
            raw_files = glob.glob(f'{mesh_modelname_directory}/*.{mesh_format}')    
            if not raw_files:
                raise ValueError(f"Didn't find any {mesh_format} files in {mesh_modelname_directory}")
            files = sorted(raw_files)
            
            temp_sample_info = sample_info[template_name]
            model_files.append((temp_sample_info, files, mesh_modelname_directory, dataset_modelname_directory))


    print('Making dataset...')
    n_jobs = os.cpu_count()
    assert max_threads != 0
    if max_threads > 0:
        n_jobs = max_threads
    output_dirs = Parallel(n_jobs=n_jobs)(
        delayed(process_one)(f[0], f[1], f[2], f[3],
                             skip_existing) for f in tqdm.tqdm(model_files))
    
    print('Done!')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_mesh_dir', action='store', dest='input_mesh_dir', required=True, help='Provide input watertight mesh directory')
    parser.add_argument('--output_data_dir', action='store', dest='output_data_dir', required=True, help='Provide output directory')
    parser.add_argument('--mesh_format', action='store', dest='mesh_format', help='Provide mesh format')
    parser.add_argument('--max_threads', action='store', type=int, dest='max_threads', help='Maximum number of threads to be used (uses all available threads by default)')
    parser.add_argument('--temp_lst', action='store', dest='temp_lst', help='Provide filter lt')
    args = parser.parse_args()

    input_mesh_dir = args.input_mesh_dir
    output_data_dir = args.output_data_dir
    mesh_format = args.mesh_format
    max_threads = args.max_threads
    temp_lst = args.temp_lst
    
    if not mesh_format:
        mesh_format = "ply"
    if not max_threads:
        max_threads = -1

    generate_flow(input_mesh_dir, output_data_dir, mesh_format, temp_lst, skip_existing=False, max_threads=max_threads)
