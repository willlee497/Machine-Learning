import random

import numpy as np
import torch

from hw6_q3 import hw6_q3_autograd as ad
from hw6_q3 import hw6_q3_nn as nn

torch.use_deterministic_algorithms(True)
torch.set_default_dtype(torch.float64)


def set_seed(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def compute_num_params(parameters):
    if isinstance(parameters, list) and isinstance(parameters[0], ad.Scalar):
        return len(parameters)
    return sum(param.numel() for param in parameters)


def compute_batch(mlp, X):
    # torch can automatically handle batched inputs
    # while we will have to do it manually
    return (
        [mlp(Xi) for Xi in X]
        if isinstance(mlp, nn.Module)
        else mlp(X).squeeze(1)
    )


def accuracy(mlp, X, y):
    logits = compute_batch(mlp, X)

    def is_correct(logit, label):
        return (logit.item() >= 0) == (label == 1)

    correct = sum(is_correct(logit, label) for logit, label in zip(logits, y))
    return correct / len(logits)


def train(
    mlp,
    loss_fn,
    optimizer,
    X,
    y,
    batch_size=8,
    num_steps=100,
):
    """Train the model

    Args:
        mlp: OurMLP or TorchMLP
        loss_fn: Our loss function or torch loss function
        optimizer (_type_): Our optimizer or torch optimizer
        X: Training data as a numpy array or torch tensor
        y (_type_): Training label as a numpy array or torch tensor
        batch_size (int, optional): Training batch size
        num_steps (int, optional): Number of training steps
        compute_acc (bool, optional): Whether to compute accuracy for the training step
    """
    set_seed()
    losses, accs = [], []
    for i in range(1, num_steps + 1):
        """
        Steps:
        - Sample a batch of X, y from ids using `random.sample`
        - Compute batch outputs
        - Compute batch loss
        - Zero gradient for all parameters
        - Backpropagate gradient
        - Perform one optimization steps
        """
        batch_ids = random.sample(range(len(X)), k=batch_size)
        batch_X = X[batch_ids]
        batch_y = y[batch_ids]

        ### YOUR IMPLEMENTATION START ###
        #compute batch outputs
        batch_outputs = compute_batch(mlp, batch_X)
        
        #compute batch loss
        if hasattr(mlp, 'zero_grad') and not hasattr(mlp, 'state_dict'):
            #my implementation - convert labels to Scalar list
            batch_y_scalars = [ad.Scalar(float(label)) for label in batch_y]
            loss = loss_fn(batch_outputs, batch_y_scalars)
        else:
            #torch implementation
            loss = loss_fn(torch.stack(batch_outputs) if isinstance(batch_outputs, list) else batch_outputs, 
                          batch_y)
        
        #zero gradients for all parameters
        if hasattr(mlp, 'zero_grad') and not hasattr(mlp, 'state_dict'):
            #my implementation
            mlp.zero_grad()
        else:
            #torch implementation
            optimizer.zero_grad()
        
        #backpropagate gradient
        loss.backward()
        
        #perform one optimization step
        optimizer.step()
        ### YOUR IMPLEMENTATION END ###

        acc = accuracy(mlp, X, y)
        print(f"Step {i:#3d} \t Loss {loss.item():#.4f} \t Accuracy {acc}")

        losses.append(loss.item())
        accs.append(acc)

    return losses, accs
