import torch

def check_statedict_equivalent(sd1, sd2) -> bool:
    """
    returns True if the torch state dicts are different and False otherwise
    """
    for key in sd1.keys():
        if not torch.isclose(sd1[key], sd2[key]).all():
            return False
    return True