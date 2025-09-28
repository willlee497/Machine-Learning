import torch

def gaussian_theta(X, y):
    """
    Compute MLE estimates of mean and variance for Gaussian Naive Bayes.

    Args:
        X (torch.Tensor): shape (N, d), features
        y (torch.Tensor): shape (N,), labels in {0,1}

    Returns:
        mu (torch.Tensor): shape (2, d), class means
        sigma2 (torch.Tensor): shape (2, d), class variances
    """
    classes = torch.unique(y)
    d = X.shape[1]
    mu = torch.zeros((len(classes), d), dtype=torch.float32)
    sigma2 = torch.zeros((len(classes), d), dtype=torch.float32)

    for idx, c in enumerate(classes):
        Xc = X[y == c]
        mu[idx] = Xc.mean(dim=0)
        sigma2[idx] = Xc.var(dim=0, unbiased=False)  # MLE variance
    return mu, sigma2

def gaussian_p(y):
    """
    Compute MLE estimate of P(Y=0).

    Args:
        y (torch.Tensor): shape (N,), labels in {0,1}

    Returns:
        float: probability P(Y=0)
    """
    return (y == 0).float().mean().item()

def gaussian_classify(mu, sigma2, p, X):
    """
    Classify test samples using Gaussian Naive Bayes (log form).

    Args:
        mu (torch.Tensor): shape (2, d), class means
        sigma2 (torch.Tensor): shape (2, d), class variances
        p (float): prior probability P(Y=0)
        X (torch.Tensor): shape (N, d), test data

    Returns:
        torch.Tensor: shape (N,), predicted labels {0,1}
    """
    n_classes, d = mu.shape
    N = X.shape[0]
    yhat = torch.zeros(N, dtype=torch.long)

    # Priors for classes 0 and 1
    logpriors = torch.tensor([p, 1.0 - p], dtype=torch.float32).log()

    # To avoid division by zero if a feature variance is 0
    eps = 1e-8
    sigma2 = sigma2.clamp_min(eps)

    for i in range(N):
        # Compute log posterior up to a constant for each class
        scores = []
        for c in range(n_classes):
            log_likelihood = -0.5 * torch.log(2 * torch.pi * sigma2[c]) \
                             -0.5 * ((X[i] - mu[c]) ** 2) / sigma2[c]
            scores.append(logpriors[c] + log_likelihood.sum())
        scores = torch.stack(scores)
        yhat[i] = torch.argmax(scores)
    return yhat

def gaussian_eval(mu, sigma2, p, X, y):
    """
    Simple evaluation helper expected by the tests.

    Returns:
        acc (float): classification accuracy in [0,1]
        yhat (torch.Tensor): predicted labels
    """
    yhat = gaussian_classify(mu, sigma2, p, X)
    acc = (yhat == y).float().mean().item()
    return acc, yhat

