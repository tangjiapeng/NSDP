import argparse
import logging
import os
import sys
import numpy as np
import torch

def load_checkpoints(model, optimizer, experiment_directory, args, device):
    model_files = [
        f for f in os.listdir(experiment_directory)
        if f.startswith("model_")
    ]
    if len(model_files) == 0:
        return
    ids = [int(f[6:]) for f in model_files]
    max_id = max(ids)
    model_path = os.path.join(
        experiment_directory, "model_{:05d}"
    ).format(max_id)
    opt_path = os.path.join(
        experiment_directory, "opt_{:05d}"
    ).format(max_id)
    if not (os.path.exists(model_path) and os.path.exists(opt_path)):
        return

    print("Loading model checkpoint from {}".format(model_path))
    model.load_state_dict(torch.load(model_path, map_location=device))
    print("Loading optimizer checkpoint from {}".format(opt_path))
    optimizer.load_state_dict(
        torch.load(opt_path, map_location=device)
    )
    args.continue_from_epoch = max_id+1


def save_checkpoints(epoch, model, optimizer, experiment_directory):
    torch.save(
        model.state_dict(),
        os.path.join(experiment_directory, "model_{:05d}").format(epoch)
    )
    torch.save(
        optimizer.state_dict(),
        os.path.join(experiment_directory, "opt_{:05d}").format(epoch)
    )




def load_best_checkpoints(model, experiment_directory, args, device):
    model_files = [
        f for f in os.listdir(experiment_directory)
        if f.startswith("modelbest_")
    ]
    if len(model_files) == 0:
        return
    ids = [f[10:] for f in model_files]
    last_id = sorted(ids)[-1]
    epoch, val_loss = int(last_id[0:5]), float(last_id[6:])
    
    model_path = os.path.join(
        experiment_directory, "modelbest_{:05d}_{:03f}"
    ).format(epoch, val_loss)
    if not os.path.exists(model_path):
        return

    print("Loading model checkpoint from {}".format(model_path))
    model.load_state_dict(torch.load(model_path, map_location=device))
    args.continue_from_epoch = epoch+1
    args.best_val_loss = val_loss

def save_best_checkpoints(epoch, model, experiment_directory, val_loss):
    torch.save(
        model.state_dict(),
        os.path.join(experiment_directory, "modelbest_{:05d}_{:03f}").format(epoch, val_loss)
    )