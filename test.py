from functools import partial
import sys, os
import argparse
import numpy as np
import math
import logging
import torch
from torch.utils.data import DataLoader

from dataset import dataset_dict
from model import build_model

from utils.training_utils import load_config
from utils.checkpoints import load_best_checkpoints
from utils.logger import StatsLogger
from utils.eval_metric import compute_evaluation_metrics
from utils.generation import generate_meshes, generate_pointclouds

def main(argv):
    parser = argparse.ArgumentParser(
        description="Train a deformation networks"
    )

    parser.add_argument(
        "config_file",
        help="Path to the file that contains the experiment configuration"
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=0,
        help="The number of processed spawned by the batch provider"
    )
    parser.add_argument(
        "--num_threads",
        type=int,
        default=4,
        help="The number of threads"
    )
    args = parser.parse_args(argv)

    # Disable trimesh's logger
    logging.getLogger("trimesh").setLevel(logging.ERROR)

    if torch.cuda.is_available():
        device = torch.device("cuda:0")
    else:
        device = torch.device("cpu")
    print("Running code on", device)

    # Parse the config file
    config = load_config(args.config_file)

    # Check if output directory exists and if it doesn't create it
    output_directory = config["experiment"]["out_dir"]

    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    # Create an experiment directory using the experiment_name
    experiment_name = config["experiment"]["name"]

    experiment_directory = os.path.join(
        output_directory,
        experiment_name
    )
    if not os.path.exists(experiment_directory):
        os.makedirs(experiment_directory)

    # Parser dataset 
    dataset_type = config['data']['type']
    Dataset = dataset_dict[dataset_type]
    test_dataset = Dataset(
        config,
        iden_split=config["test"]["iden_split"],
        motion_split=config["test"]["motion_split"],
        load_mesh=config["test"]["load_mesh"], 
        num_sampled_pairs=config["test"]["num_sampled_pairs"]
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=config["test"].get("batch_size", 1),
        num_workers=args.num_workers,
        collate_fn=test_dataset.collate_fn,
        shuffle=False,
    )
    print("Loaded {} test deformation pairs".format( len(test_dataset) ))


    # Build the network architecture to be used for training
    weight_file = config["test"].get("weight_file")
    model, _, _, test_on_batch = build_model(
        config, weight_file, device=device
    )

    # Log the stats to a file
    StatsLogger.instance().add_output_file(open(
        os.path.join(experiment_directory, "{}.txt".format(config["test"]["motion_split"]) ),
        "w"
    ))

    if config['test']['generate_mesh']:
        generation_mesh_directory = os.path.join(
            output_directory,
            experiment_name,
            config["test"]["motion_split"],
            config['test']['mesh_folder'],
        )
        if not os.path.exists(generation_mesh_directory):
            os.makedirs(generation_mesh_directory)
        print("Save generated meshes in {}".format(generation_mesh_directory))

    if config['test']['generate_pointcloud']:
        generation_pointcloud_directory = os.path.join(
            output_directory,
            experiment_name,
            config["test"]["motion_split"],
            config['test']['pointcloud_folder'],
        )
        if not os.path.exists(generation_pointcloud_directory):
            os.makedirs(generation_pointcloud_directory)
        print("Save generated pointclouds in {}".format(generation_pointcloud_directory))

    # Do the inference
    print("====> Inference / Test ====>")
    model.eval()
    for b, sample in enumerate(test_loader):
        # Move everything to device
        for k, v in sample.items():
            sample[k] = v.to(device)
        batch_loss, out_dict = test_on_batch(model, sample, config, compute_loss=True)
        
        # compute evaluation metrics and obtain statistics
        eval_dict = compute_evaluation_metrics(out_dict)
        for k, v in eval_dict.items():
            #print(k, v)
            if v <= 1.0:
                StatsLogger.instance()[k].value = v
        StatsLogger.instance().print_progress(-1, b+1, batch_loss)

        # get the deformation pair_info of b-th test data samples
        sample_idx = out_dict["index"].item()
        meta_data = test_dataset.get_metadata(sample_idx)

        # generate source / canonical / target meshes and/or point clouds
        if config['test']['generate_mesh']:
            generate_meshes(generation_mesh_directory, out_dict, meta_data, config['test']['mesh_format'], vert_pred_color=True)

        if config['test']['generate_pointcloud']:
            generate_pointclouds(generation_pointcloud_directory, out_dict, meta_data, config['test']['pointcloud_format'])

    StatsLogger.instance().clear()
    print("====> Inference / Test ====>")        

if __name__ == "__main__":
    main(sys.argv[1:])