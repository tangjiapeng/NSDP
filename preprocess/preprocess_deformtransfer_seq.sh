#!/bin/sh


input_mesh_dir="/cluster/balrog/jtang/DeformationTransfer_processed"
output_data_dir="/cluster_HDD/umoja/jtang/DeformationTransfer_processed_dataset"
max_threads=12
mesh_format="obj"


filter_lst="../data/splits/deformtransfer/test_unseen_identities.lst"
python generate_dataset_deformtransfer_seq.py --input_mesh_dir="${input_mesh_dir}" --output_data_dir="${output_data_dir}" --max_threads=${max_threads} \
      --mesh_format "${mesh_format}" #--filter_lst "${filter_lst}"


temp_lst="../data/splits/deformtransfer/identity_unseen.lst"
python generate_dataset_deformtransfer_surfaceflow.py --input_mesh_dir="${input_mesh_dir}" --output_data_dir="${output_data_dir}" --max_threads=${max_threads} \
      --mesh_format "${mesh_format}"  --temp_lst "${temp_lst}"


temp_lst="../data/splits/deformtransfer/identity_unseen.lst"
python generate_dataset_deformtransfer_spaceflow.py --input_mesh_dir="${input_mesh_dir}" --output_data_dir="${output_data_dir}" --max_threads=${max_threads} \
     --mesh_format "${mesh_format}"  --temp_lst "${temp_lst}"
