import torch
import numpy as np

__all__ = ["gaussian_theta", "gaussian_p", "gaussian_classify", "gaussian_eval"]

def _to_tensor(a):
    if isinstance(a, torch.Tensor):
        return a
    return torch.as_tensor(a, dtype=torch.float32)

def gaussian_theta(X, y):
    """
    MLE estimates for Gaussian NB per class/feature.
    Returns:
        mu: (2, d)  class means in 0/1 order
        sigma2: (2, d) class variances in 0/1 order (MLE, unbiased=False)
    """
    X = _to_tensor(X); y = _to_tensor(y).long()
    d = X.shape[1]
    mu = torch.zeros((2, d), dtype=torch.float32)
    sigma2 = torch.zeros((2, d), dtype=torch.float32)
    for c in (0, 1):
        Xc = X[y == c]
        mu[c] = Xc.mean(dim=0)
        sigma2[c] = Xc.var(dim=0, unbiased=False)
    return mu, sigma2

def gaussian_p(y):
    y = _to_tensor(y)
    return float((y == 0).float().mean().item())

def gaussian_classify(mu, sigma2, p, X):
    """
    If X is np.ndarray, returns np.ndarray yhat; else a torch.LongTensor.
    """
    x_was_numpy = isinstance(X, np.ndarray)
    mu = _to_tensor(mu); sigma2 = _to_tensor(sigma2); X = _to_tensor(X)

    eps = 1e-8
    sigma2 = sigma2.clamp_min(eps)                      # (2, d)
    logpriors = torch.log(torch.tensor([p, 1.0 - p], dtype=torch.float32))  # (2,)

    # vectorized over N: X -> (N,1,d), mu/sigma2 -> (1,2,d)
    X3 = X.unsqueeze(1)                                  # (N,1,d)
    const = -0.5 * torch.log(2.0 * torch.pi * sigma2)    # (2,d)
    quad  = -0.5 * ((X3 - mu) ** 2) / sigma2             # (N,2,d)
    log_lik = (const + quad).sum(dim=2)                  # (N,2)
    scores  = log_lik + logpriors.view(1, 2)             # (N,2)
    yhat = torch.argmax(scores, dim=1).long()            # (N,)

    if x_was_numpy:
        return yhat.numpy()
    return yhat

def gaussian_eval(mu, sigma2, p, X, y):
    """
    Helper the autograder imports by name from hw2_q2.
    Returns:
        acc: float in [0,1]
        yhat: (N,) same array/tensor type as X
    """
    yhat = gaussian_classify(mu, sigma2, p, X)
    yt = _to_tensor(y).long()
    yh = _to_tensor(yhat).long()
    acc = float((yh == yt).float().mean().item())
    # Preserve numpy type if X was numpy
    if isinstance(X, np.ndarray) and isinstance(yhat, np.ndarray):
        return acc, yhat
    return acc, yh
