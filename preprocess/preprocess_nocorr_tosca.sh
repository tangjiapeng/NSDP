#!/bin/sh
input_mesh_dir="/cluster/balrog/jtang/TOSCA"
output_data_dir="/cluster_HDD/umoja/jtang/TOSCA_dataset"
max_threads=12
mesh_format="off"

filter_lst="../data/splits/tosca/val_unseen_identities.lst"
python generate_dataset_nocorr.py --input_mesh_dir="${input_mesh_dir}" --output_data_dir="${output_data_dir}" --max_threads=${max_threads} \
      --mesh_format "${mesh_format}" #--filter_lst "${filter_lst}"