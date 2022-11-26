#!/bin/sh


input_mesh_dir="/cluster_HDD/umoja/jtang/DeformingThings4D_processed/animals"
output_data_dir="/cluster_HDD/umoja/jtang/DeformingThings4D_processed_dataset/animals"
max_threads=12
mesh_format="obj"
interval=3


python generate_dataset_deform4d_seq.py --input_mesh_dir="${input_mesh_dir}" --output_data_dir="${output_data_dir}" --max_threads=${max_threads} \
     --mesh_format "${mesh_format}" --interval ${interval} 


temp_lst="../data/splits/deform4d/identity_all.lst"
python generate_dataset_deform4d_spaceflow.py --input_mesh_dir="${input_mesh_dir}" --output_data_dir="${output_data_dir}" --max_threads=${max_threads} \
     --mesh_format "${mesh_format}"  --interval ${interval} --temp_lst "${temp_lst}"


temp_lst="../data/splits/deform4d/identity_all.lst"
python generate_dataset_deform4d_surfaceflow.py --input_mesh_dir="${input_mesh_dir}" --output_data_dir="${output_data_dir}" --max_threads=${max_threads} \
     --mesh_format "${mesh_format}"  --interval ${interval}  --temp_lst "${temp_lst}"
