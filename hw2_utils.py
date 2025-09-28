import torch

def gaussian_dataset(split="train"):
    """
    Load Gaussian dataset (stub). 
    Replace with actual dataset logic if provided by instructors.
    
    Args:
        split (str): "train" or "test"
    
    Returns:
        X (torch.Tensor): features
        y (torch.Tensor): labels
    """
    if split == "train":
        X = torch.randn(100, 2)
        y = (X[:,0] + X[:,1] > 0).long()
    else:
        X = torch.randn(50, 2)
        y = (X[:,0] + X[:,1] > 0).long()
    return X, y
