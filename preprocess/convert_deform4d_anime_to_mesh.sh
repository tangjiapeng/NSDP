#!/bin/sh

in_folder="/cluster/balrog/jtang/DeformingThings4D/animals"
mesh_folder="/cluster_HDD/umoja/jtang/DeformingThings4D_processed"
n_proc=12
in_ext='anime'
out_ext='obj'

python convert_deform4d_anime_to_mesh.py ${in_folder} --mesh_folder="${mesh_folder}" --n_proc=${n_proc} --in_ext=${in_ext} --out_ext=${out_ext} --overwrite