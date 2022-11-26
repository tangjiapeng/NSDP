from functools import partial
import sys, os
import argparse
import numpy as np
import math
import logging
import torch
from torch.utils.data import DataLoader

from tensorboardX import SummaryWriter
from dataset import dataset_dict
from model import build_model, optimizer_factory
from model.learningrate import adjust_learning_rate, get_learning_rates, print_num_parameters

from utils.training_utils import save_experiment_params, load_config
from utils.checkpoints import load_checkpoints, save_checkpoints, load_best_checkpoints, save_best_checkpoints
from utils.logger import StatsLogger, WandB

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
    parser.add_argument(
        "--seed",
        type=int,
        default=27,
        help="Seed for the PRNG"
    )
    parser.add_argument(
        "--continue_from_epoch",
        default=0,
        type=int,
        help="Continue training from epoch (default=0)"
    )
    parser.add_argument(
        "--best_val_loss",
        type=float,
        default=9999999999999,
        help="The default value for the best val loss"
    )
    parser.add_argument(
        "--with_wandb_logger",
        action="store_true",
        help="Use wandB for logging the training progress"
    )
    args = parser.parse_args(argv)

    # Disable trimesh's logger
    logging.getLogger("trimesh").setLevel(logging.ERROR)

    # Set the random seed
    np.random.seed(args.seed)
    torch.manual_seed(np.random.randint(np.iinfo(np.int32).max))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(np.random.randint(np.iinfo(np.int32).max))

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

    # Save the parameters of this run to a file
    save_experiment_params(args, experiment_name, experiment_directory)
    print("Save experiment statistics in {}".format(experiment_directory))
    
    # Parser dataset
    dataset_type = config['data']['type']
    Dataset = dataset_dict[dataset_type]
    train_dataset = Dataset(
        config,
        iden_split=config["training"]["iden_split"],
        motion_split=config["training"]["motion_split"],
        load_mesh=config["training"]["load_mesh"], 
        num_sampled_pairs=config["training"]["num_sampled_pairs"]
    )
    validation_dataset = Dataset(
        config,
        iden_split=config["validation"]["iden_split"],
        motion_split=config["validation"]["motion_split"],
        load_mesh=config["validation"]["load_mesh"], 
        num_sampled_pairs=config["validation"]["num_sampled_pairs"]
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=config["training"].get("batch_size", 16),
        num_workers=args.num_workers,
        collate_fn=train_dataset.collate_fn,
        shuffle=True
    )
    print("Loaded {} training deformation pairs".format( len(train_dataset) ))

    val_loader = DataLoader(
        validation_dataset,
        batch_size=config["validation"].get("batch_size", 1),
        num_workers=args.num_workers,
        collate_fn=validation_dataset.collate_fn,
        shuffle=False
    )
    print("Loaded {} validation deformation pairs".format( len(validation_dataset) ))

    # Get the weight file to initilize the networks before training
    weight_file = config["training"].get("weight_file", None)
    weight_forward_file = config["training"].get("weight_forward_file", None)
    weight_backward_file = config["training"].get("weight_backward_file", None)

    # Build the network architecture to be used for training
    model, train_on_batch, validate_on_batch, _ = build_model(
        config, weight_file, weight_forward_file, weight_backward_file, device=device
    )
    # Count trainable parameters.
    print_num_parameters(model)

    # Build an optimizer object to compute the gradients of the parameters
    lr_scheduler, optimizer = optimizer_factory(config["training"], model.parameters())

    # Load the checkpoints if they exist in the experiment directory
    # Load the best_val_loss and the corresponding model
    load_best_checkpoints(model, experiment_directory, args, device)
    # Then load the latest model
    load_checkpoints(model, optimizer, experiment_directory, args, device)
    
    # Initialize the logger
    if args.with_wandb_logger:
        WandB.instance().init(
            config,
            model=model,
            project=config["logger"].get(
                "project", "NSDP"
            ),
            name=experiment_name,
            watch=False,
            log_frequency=10
        )

    # Log the stats to a file
    StatsLogger.instance().add_output_file(open(
        os.path.join(experiment_directory, "stats.txt"),
        "w"
    ))

    epochs = config["training"].get("epochs", 1000)
    save_every = config["training"].get("save_frequency", 20)
    val_every = config["validation"].get("frequency", 10)


    # Do the training
    for i in range(args.continue_from_epoch, epochs):
        # adjust learning rate
        adjust_learning_rate(lr_scheduler, optimizer, i)

        model.train()
        for b, sample in enumerate(train_loader):
            # Move everything to device
            for k, v in sample.items():
                sample[k] = v.to(device)
            batch_loss = train_on_batch(model, optimizer, sample, config)
            StatsLogger.instance().print_progress(i+1, b+1, batch_loss)

        if (i % save_every) == 0:
            save_checkpoints(
                i,
                model,
                optimizer,
                experiment_directory,
            )
        StatsLogger.instance().clear()

        if i % val_every == 0 and i > 0:
            print("====> Validation Epoch ====>")
            model.eval()
            for b, sample in enumerate(val_loader):
                # Move everything to device
                for k, v in sample.items():
                    sample[k] = v.to(device)
                batch_loss = validate_on_batch(model, sample, config)
                StatsLogger.instance().print_progress(-1, b+1, batch_loss)

            val_loss_i = StatsLogger.instance()._loss.value
            if  val_loss_i < args.best_val_loss:
                save_best_checkpoints(
                    i,
                    model,
                    experiment_directory,
                    val_loss_i,
                )
                args.best_val_loss = val_loss_i
            StatsLogger.instance().clear()
            print("====> Validation Epoch ====>")        

if __name__ == "__main__":
    main(sys.argv[1:])