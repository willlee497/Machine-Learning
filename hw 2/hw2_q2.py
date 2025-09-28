import torch

def gaussian_theta(X, y):
    """
    MLE estimates for Gaussian NB per class/feature.
    Returns:
        mu: (2, d)  class means in 0/1 order
        sigma2: (2, d) class variances in 0/1 order (MLE, unbiased=False)
    """
    d = X.shape[1]
    mu = torch.zeros((2, d), dtype=torch.float32)
    sigma2 = torch.zeros((2, d), dtype=torch.float32)
    for c in (0, 1):
        Xc = X[y == c]
        mu[c] = Xc.mean(dim=0)
        sigma2[c] = Xc.var(dim=0, unbiased=False)
    return mu, sigma2

def gaussian_p(y):
    """MLE P(Y=0)."""
    return (y == 0).float().mean().item()

def gaussian_classify(mu, sigma2, p, X):
    """
    Gaussian NB classification in log-space.
    Args:
        mu, sigma2: (2, d) tensors
        p: float P(Y=0)
        X: (N, d) tensor
    Returns:
        yhat: (N,) long {0,1}
    """
    eps = 1e-8
    sigma2 = sigma2.clamp_min(eps)
    logpriors = torch.tensor([p, 1.0 - p], dtype=torch.float32).log()

    const = -0.5 * torch.log(2.0 * torch.pi * sigma2)  # (2, d)
    N = X.shape[0]
    yhat = torch.zeros(N, dtype=torch.long)
    for i in range(N):
        x = X[i]
        log_lik = const - 0.5 * ((x - mu) ** 2) / sigma2   # (2, d)
        scores = log_lik.sum(dim=1) + logpriors           # (2,)
        yhat[i] = torch.argmax(scores)
    return yhat

def gaussian_eval(mu, sigma2, p, X, y):
    """
    Helper the autograder imports by name.
    Returns:
        acc: float in [0,1]
        yhat: (N,) long
    """
    yhat = gaussian_classify(mu, sigma2, p, X)
    acc = (yhat == y).float().mean().item()
    return acc, yhat
