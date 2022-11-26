import argparse
import numpy as np
import os
import glob
import sys
from multiprocessing import Pool
from functools import partial
import trimesh
import math

parser = argparse.ArgumentParser('Sample a watertight mesh.')
parser.add_argument('in_folder', type=str, 
                    help='Path to input watertight meshes.')
parser.add_argument('--mesh_folder', type=str,
                    help='Output path for mesh.')
parser.add_argument('--in_ext', type=str, default='anime',
                    help='Extensions for input meshes.')
parser.add_argument('--out_ext', type=str, default='obj',
                    help='Extensions for ouput meshes.')
parser.add_argument('--n_proc', type=int, default=0,
                    help='Number of processes to use.')
parser.add_argument('--overwrite', action='store_true',
                    help='Whether to overwrite output.')
    
def main(args):
    #input_files = glob.glob(os.path.join(args.in_folder, '*.anime'))
    modelnames = sorted(os.listdir(os.path.join(args.in_folder)))
    input_files = []
    for mod in modelnames:
        mod_folder = os.path.join(args.in_folder, mod)
        files = glob.glob(os.path.join(mod_folder, '*.%s'%args.in_ext))
        input_files.extend(files)
    print(input_files)
    print(len(input_files))
    if args.n_proc != 0:
        with Pool(args.n_proc) as p:
            p.map(partial(process_path, args=args), input_files)
    else:
        for p in input_files:
            process_path(p, args)


def process_path(in_path, args):
    in_file = os.path.basename(in_path)
    catname = in_path.split('/')[-3]
    modelname = in_path.split('/')[-2]
    filename = os.path.splitext(in_file)[0]
    
    export_mesh(in_path, catname, modelname, args) 

def anime_read(filename):
    """
    filename: path of .anime file
    return:
        nf: number of frames in the animation
        nv: number of vertices in the mesh (mesh topology fixed through frames)
        nt: number of triangle face in the mesh
        vert_data: vertice data of the 1st frame (3D positions in x-y-z-order)
        face_data: riangle face data of the 1st frame
        offset_data: 3D offset data from the 2nd to the last frame
    """
    f = open(filename, 'rb')
    nf = np.fromfile(f, dtype=np.int32, count=1)[0]
    nv = np.fromfile(f, dtype=np.int32, count=1)[0]
    nt = np.fromfile(f, dtype=np.int32, count=1)[0]
    vert_data = np.fromfile(f, dtype=np.float32, count=nv * 3)
    face_data = np.fromfile(f, dtype=np.int32, count=nt * 3)
    offset_data = np.fromfile(f, dtype=np.float32, count=-1)
    '''check data consistency'''
    if len(offset_data) != (nf - 1) * nv * 3:
        raise ("data inconsistent error!", filename)
    vert_data = vert_data.reshape((-1, 3))
    face_data = face_data.reshape((-1, 3))
    offset_data = offset_data.reshape((nf - 1, nv, 3))
    return nf, nv, nt, vert_data, face_data, offset_data


def export_mesh(in_file, catname, modelname, args):
    output_filefolder = os.path.join(args.mesh_folder, catname, modelname)
    if not os.path.exists(output_filefolder):
        os.makedirs(output_filefolder)
    
    nf, nv, nt, vert_data, face_data, offset_data = anime_read(in_file)
    print('nframes :', nf, 'output_folder :', output_filefolder)
    print('nverts :', nv, 'nfaces :', nt)
    print('offset data :', offset_data.shape)
    for i in range(nf):
        filename = os.path.join(output_filefolder, '%04d.%s'%(i, args.out_ext) )
        if not args.overwrite and os.path.exists(filename):
            print('Mesh already exist: %s' % filename)
            return
        
        if i == 0:
            mesh = trimesh.Trimesh(vert_data, face_data, process=False)
            mesh.export(filename)
        else:
            verts_offset_i = offset_data[i-1, :, :]
            mesh = trimesh.Trimesh(vert_data+verts_offset_i, face_data, process=False)
            mesh.export(filename)
        print('Writing mesh : %s' % filename)
    
if __name__ == '__main__':
    args = parser.parse_args()
    main(args)