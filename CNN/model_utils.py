"""
Utility functions for working with pytorch models and state dicts
"""
import torch
from collections import OrderedDict


def is_distributed_model(state_dict):
    """
    determines if the state dict is from a model trained on distributed GPUs
    """
    return all(k.startswith("module.") for k in state_dict.keys())


def strip_distributed_keys(state_dict):
    """
    if the state_dict was trained across multiple GPU's then the state_dict
    keys are prefixed with 'module.', which will not match the keys
    of the new model, when we try to load the model state
    """
    assert is_distributed_model(state_dict)
    new_state_dict = OrderedDict()
    for key, value in state_dict.items():
        key = key[7:]
        new_state_dict[key] = value
    return new_state_dict


def load_model_weights(model, path_to_state_dict, USE_GPU=True, strip_keys=True):
    """
    Load a model with a given state (pre-trained weights)

    Parameters:
    -----------
    model: pytorch model
    path_to_state_dict: string

    Returns:
    ---------
    pytorch model with weights loaded to state_dict
    """
    if USE_GPU:
        model_state = torch.load(path_to_state_dict)
    else:
        # need to map storage loc to cpu
        model_state = torch.load(
            path_to_state_dict, map_location=lambda storage, loc: storage
        )
    if is_distributed_model(model_state) and strip_keys:
        model_state = strip_distributed_keys(model_state)
    model.load_state_dict(model_state)
    model.eval()
    if USE_GPU:
        model = model.cuda()
    return model
