#!/bin/sh
input_mesh_dir="/cluster/balrog/jtang/dog_barc_recon"
output_data_dir="/cluster_HDD/umoja/jtang/dogrec_dataset"
max_threads=4
mesh_format="obj"

filter_lst="../data/splits/dogrec/test_unseen_identities.lst"
python generate_dataset_nocorr.py --input_mesh_dir="${input_mesh_dir}" --output_data_dir="${output_data_dir}" --max_threads=${max_threads} \
      --mesh_format "${mesh_format}" #--filter_lst "${filter_lst}"