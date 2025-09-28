import torch

def sigmoid(z):
    """Sigmoid activation."""
    return 1 / (1 + torch.exp(-z))

def compute_cost(X, y, w, b, reg=None, lam=0.0):
    """
    Compute binary cross-entropy cost with optional regularization.
    
    Args:
        X (torch.Tensor): shape (N, d), features
        y (torch.Tensor): shape (N,), labels in {0,1}
        w (torch.Tensor): weights, shape (d,1)
        b (torch.Tensor): bias scalar
        reg (str): "l1" or "l2" or None
        lam (float): regularization strength
    
    Returns:
        float: cost
    """
    m = X.shape[0]
    z = X @ w + b
    A = sigmoid(z)
    eps = 1e-8
    cost = -(y*torch.log(A+eps) + (1-y)*torch.log(1-A+eps)).mean()

    if reg == "l2":
        cost += lam / (2*m) * (w**2).sum()
    elif reg == "l1":
        cost += lam / (2*m) * w.abs().sum()
    return cost

def compute_gradients(X, y, w, b, reg=None, lam=0.0):
    """
    Compute gradients of logistic regression.
    """
    m = X.shape[0]
    z = X @ w + b
    A = sigmoid(z)
    dw = (X.T @ (A - y)) / m
    db = (A - y).mean()

    if reg == "l2":
        dw += (lam/m) * w
    elif reg == "l1":
        dw += (lam/m) * torch.sign(w)
    return dw, db

def train(X, y, alpha=0.1, epochs=1000, reg=None, lam=0.0, batch_size=None):
    """
    Train logistic regression model using GD variants.
    
    Args:
        X (torch.Tensor): shape (N, d)
        y (torch.Tensor): shape (N,)
        alpha (float): learning rate
        epochs (int): number of passes
        reg (str): "l1" or "l2" or None
        lam (float): regularization strength
        batch_size (int): if None, batch GD; else mini-batch / SGD
    
    Returns:
        (w, b): trained parameters
    """
    m, d = X.shape
    w = torch.zeros(d, 1)
    b = torch.zeros(1)
    y = y.view(-1,1).float()

    for epoch in range(epochs):
        if batch_size is None:  # Batch GD
            dw, db = compute_gradients(X, y, w, b, reg, lam)
            w -= alpha * dw
            b -= alpha * db
        else:  # Mini-batch or SGD
            for i in range(0, m, batch_size):
                Xb = X[i:i+batch_size]
                yb = y[i:i+batch_size]
                dw, db = compute_gradients(Xb, yb, w, b, reg, lam)
                w -= alpha * dw
                b -= alpha * db
    return w, b
