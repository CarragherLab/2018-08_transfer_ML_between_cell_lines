"""
Utility functions for working with pytorch models and state dicts
"""

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
