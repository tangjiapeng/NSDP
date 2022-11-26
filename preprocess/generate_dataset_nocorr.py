import os
from posixpath import basename
from random import randrange
from select import select
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


def process_one(mesh_path, idx, mesh_directory, dataset_directory, skip_existing):
    """Processes a single mesh, adding it to the dataset."""
    name = os.path.basename(mesh_path)
    name, extension = os.path.splitext(name)
    
    # define the generate filename using sequence_idx
    name = '%04d'%idx

    valid_extensions = ['.obj', '.off', '.ply'] 
    if extension not in valid_extensions:
        print(mesh_path, extension)
        raise ValueError(f'File with unsupported extension {extension} found: {mesh_path}.'
                         f' Only {valid_extensions} are supported.')
    output_dir = f'{dataset_directory}/{name}/'
    scripts_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "preprocess", "others")
    external_root_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "external")

    # This is the last file the processing writes, if it already exists the
    # example has already been processed.
    if not skip_existing or not os.path.isfile(f'{output_dir}/orig_to_gaps.txt'):
        print(f'{scripts_dir}/process_mesh_local.sh {mesh_path} {output_dir} {external_root_dir}')
        sp.check_output(
            f'{scripts_dir}/process_mesh_local.sh {mesh_path} {output_dir} {external_root_dir}',
            shell=True)
    else:
        print(f'Skipping shell script processing for {output_dir},' ' the output already exists.')

    return output_dir


def generate_meshes(mesh_directory, dataset_directory, mesh_format, skip_existing=True, max_threads=-1, filter_lst=None):
    if filter_lst is not None:
        select_modnames = open(filter_lst, 'r').readlines()
        select_modnames = [f.strip() for f in select_modnames]
    
    # Make the directories first because it's not threadsafe and also might fail.
    print('Creating directories...')
    if not os.path.isdir(f'{dataset_directory}'):
        os.makedirs(f'{dataset_directory}')

    model_files = []
    mesh_directories = []
    dataset_directories = []
    file_idx_names = []
    
    for f in sorted(os.listdir(mesh_directory)):
        if os.path.isdir(os.path.join(mesh_directory, f)): 
            if filter_lst is not None:
                if f not in select_modnames:
                    print(f"{f} is not in selected clean models")
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
            file_idx_names += [i for i in range(len(files))]
            
            # append files, mesh_directory, and dataset directory
            model_files += files
            mesh_directories += [mesh_modelname_directory for _ in range(len(files))]
            dataset_directories += [dataset_modelname_directory for _ in range(len(files))]
            print(len(files), files)

    n_jobs = os.cpu_count()
    assert max_threads != 0
    if max_threads > 0:
        n_jobs = max_threads
    output_dirs = Parallel(n_jobs=n_jobs)(
        delayed(process_one)(model_files[idx], file_idx_names[idx], mesh_directories[idx], dataset_directories[idx],
                             skip_existing) for idx in tqdm.tqdm(range(len(model_files))))
    
    print('Done!')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_mesh_dir', action='store', dest='input_mesh_dir', required=True, help='Provide input watertight mesh directory')
    parser.add_argument('--output_data_dir', action='store', dest='output_data_dir', required=True, help='Provide output directory')
    parser.add_argument('--mesh_format', action='store', dest='mesh_format', help='Provide mesh format')
    parser.add_argument('--max_threads', action='store', type=int, dest='max_threads', help='Maximum number of threads to be used (uses all available threads by default)')
    parser.add_argument('--filter_lst', action='store', dest='filter_lst', help='Provide filter lt')

    args = parser.parse_args()

    input_mesh_dir = args.input_mesh_dir
    output_data_dir = args.output_data_dir
    mesh_format = args.mesh_format
    max_threads = args.max_threads
    filter_lst = args.filter_lst
    
    if not mesh_format:
        mesh_format = "ply"
    if not max_threads:
        max_threads = -1
    if not filter_lst:
        filter_lst=None

    generate_meshes(input_mesh_dir, output_data_dir, mesh_format, max_threads=max_threads, filter_lst=filter_lst)