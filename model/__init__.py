import torch
import torch.nn as nn
import torch.nn.functional as functional
import numpy as np
import torch.nn.functional as F
from model.deformation_networks import Deformation_Networks, train_on_batch_with_cano, validate_on_batch_with_cano, test_on_batch_with_cano
from model.flow_arbitrary import FlowArbitrary, train_on_batch_with_arbitrary, validate_on_batch_with_arbitrary, test_on_batch_with_arbitrary
from model.learningrate import StepLearningRateSchedule

def optimizer_factory(config, parameters):
    """Based on the provided config create the suitable optimizer."""
    optimizer = config.get("optimizer", "Adam")
    lr_init = config.get("lr", 1e-3)
    lr_step  = config.get("lr_step", 100)
    lr_decay = config.get("lr_decay", 0.1)
    momentum = config.get("momentum", 0.9)
    weight_decay = config.get("weight_decay", 0.0)

    # Set up LearningRateSchedule
    lr_schedule = StepLearningRateSchedule({
            "type": "step",
            "initial": lr_init,
            "interval": lr_step,
            "factor": lr_decay,
        },)

    # Set up optimizer.
    if optimizer == "SGD":
        return lr_schedule, torch.optim.SGD([
            {
             'params': parameters, 'lr':lr_schedule.get_learning_rate(0), 'momentum':momentum, 'weight_decay':weight_decay
            }, 
        ])
    elif optimizer == "Adam":
        return lr_schedule, torch.optim.Adam([
            {
             'params': parameters, 'lr':lr_schedule.get_learning_rate(0), 'weight_decay':weight_decay
            }, 
        ])
    else:
        raise NotImplementedError()

def build_model(
    config,
    weight_file=None,
    weight_forward_file=None,
    weight_backward_file=None,
    device="cpu"):
    
    model_type = config["model"]["type"]

    if model_type == "forward":
        train_on_batch = train_on_batch_with_cano
        validate_on_batch = validate_on_batch_with_cano
        test_on_batch = test_on_batch_with_cano
        model = Deformation_Networks(
            config, no_input_corr=False
        )
    elif model_type == "backward":
        train_on_batch = train_on_batch_with_cano
        validate_on_batch = validate_on_batch_with_cano
        test_on_batch = test_on_batch_with_cano
        model = Deformation_Networks(
            config, no_input_corr=True
        )
    elif model_type == "arbitrary":
        train_on_batch = train_on_batch_with_arbitrary
        validate_on_batch = validate_on_batch_with_arbitrary
        test_on_batch = test_on_batch_with_arbitrary
        model_canonicalize = Deformation_Networks(
            config, no_input_corr=True
        )
        model_deform = Deformation_Networks(
            config, no_input_corr=False
        )
        model = FlowArbitrary(config, model_canonicalize, model_deform)
        
    else:
        raise NotImplementedError()
    
    # when training deformation_arbitrary models, we firstly load the forward and backward deformation models 
    if model_type == "arbitrary":
        if weight_forward_file is not None:
            print("Loading weight forward file from {}".format(weight_forward_file))
            try:
                model_deform.load_state_dict(
                    torch.load(weight_forward_file, map_location=device)
                )
            except:
                model_deform.load_state_dict(
                    torch.load(weight_forward_file, map_location=device)["model_state_dict"]
                )
            model.model_deform = model_deform
        if weight_backward_file is not None:
            print("Loading weight backward file from {}".format(weight_backward_file))
            try:
                model_canonicalize.load_state_dict(
                    torch.load(weight_backward_file, map_location=device)
                )
            except:
                model_canonicalize.load_state_dict(
                    torch.load(weight_backward_file, map_location=device)["model_state_dict"]
                )
            model.model_canonicalize = model_canonicalize

    # Check whether there is a weight file provided to continue training from
    if weight_file is not None:
        print("Loading weight file from {}".format(weight_file))
        try:
            model.load_state_dict(
                torch.load(weight_file, map_location=device)
            )
        except:
            model.load_state_dict(
                torch.load(weight_file, map_location=device)["model_state_dict"]
            )
    model.to(device)
    return model, train_on_batch, validate_on_batch, test_on_batch
