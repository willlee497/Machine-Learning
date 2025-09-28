import numpy
import torch

## You only need to complete the two functions.
def numpy_squares(k):
    """return (1, 4, 9, ... , k^2) as a numpy array"""
    arr = numpy.arange(1, k + 1)
    return arr * arr

def torch_squares(k):
    """return (1, 4, 9, ... , k^2) as a torch array"""
    t = torch.arange(1, k + 1)
    return t * t