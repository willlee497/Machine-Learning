import hw3_utils
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt

def gram_matrix(xa, xb, kernel):
    """
    Build pairwise Gram matrix using a scalar-returning kernel:
    K_ij = kernel(xa[i], xb[j]).
    """
    N, M = xa.shape[0], xb.shape[0]
    K = torch.empty((N, M), dtype=xa.dtype, device=xa.device)
    for i in range(N):
        for j in range(M):
            K[i, j] = kernel(xa[i], xb[j])  # hw3_utils.poly/rbf yield a scalar
    return K

def svm_solver(x_train, y_train, lr, num_iters,
               kernel=hw3_utils.poly(degree=1), c=None):
    '''
    Computes an SVM given a training set, training labels, the number of
    iterations to perform projected gradient descent, a kernel, and a trade-off
    parameter for soft-margin SVM.

    Arguments:
        x_train: 2d tensor with shape (N, d).
        y_train: 1d tensor with shape (N,), whose elememnts are +1 or -1.
        lr: The learning rate.
        num_iters: The number of gradient descent steps.
        kernel: The kernel function.
           The default kernel function is 1 + <x, y>.
        c: The trade-off parameter in soft-margin SVM.
           The default value is None, referring to the basic, hard-margin SVM.

    Returns:
        alpha: a 1d tensor with shape (N,), denoting an optimal dual solution.
               Initialize alpha to be 0.
               Return alpha.detach() could possibly help you save some time
               when you try to use alpha in other places.

    Note that if you use something like alpha = alpha.clamp(...) with
    torch.no_grad(), you will have alpha.requires_grad=False after this step.
    You will then need to use alpha.requires_grad_().
    Alternatively, use in-place operations such as clamp_().
    '''
    x = x_train
    y = y_train.to(x.dtype).view(-1)  # (+1/-1) as float
    N = x.shape[0]

    # Correct N×N Gram and Q = Y K Y
    K = gram_matrix(x, x, kernel)
    Q = (y[:, None] * y[None, :]) * K

    alpha = torch.zeros(N, dtype=x.dtype, device=x.device)
    one = torch.ones(N, dtype=x.dtype, device=x.device)

    for _ in range(num_iters):
        grad = Q @ alpha - one  # ∇(0.5 a^T Q a − 1^T a)
        with torch.no_grad():
            alpha -= lr * grad
            if c is None:
                alpha.clamp_(min=0.0)               # hard margin
            else:
                alpha.clamp_(min=0.0, max=float(c)) # soft margin

    return alpha.detach()

def svm_predictor(alpha, x_train, y_train, x_test,
                  kernel=hw3_utils.poly(degree=1), c=None):
    '''
    Returns the kernel SVM's predictions for x_test using the SVM trained on
    x_train, y_train with computed dual variables alpha.

    Arguments:
        alpha: 1d tensor with shape (N,), denoting an optimal dual solution.
        x_train: 2d tensor with shape (N, d), denoting the training set.
        y_train: 1d tensor with shape (N,), whose elements are +1 or -1.
        x_test: 2d tensor with shape (M, d), denoting the test set.
        kernel: The kernel function.
           The default kernel function is 1 + <x, y>.

    Return:
        A 1d tensor with shape (M,), the outputs of SVM on the test set.
    '''
    x_tr, x_te = x_train, x_test
    y = y_train.to(x_tr.dtype).view(-1)
    a = alpha.to(x_tr.dtype).view(-1)
    N = x_tr.shape[0]

    eps = 1e-10
    if c is None:
        sv_mask = (a > eps)
    else:
        sv_mask = (a > eps) & (a < (float(c) - eps))

    if not torch.any(sv_mask):
        sv_mask = (a > eps)

    if torch.any(sv_mask):
        eligible = torch.nonzero(sv_mask, as_tuple=False).view(-1)
        i_star = int(eligible[a[eligible].argmin().item()])
    else:
        i_star = int(torch.argmax(a).item())

    # b via KKT on i*
    K_train_sv = gram_matrix(x_tr, x_tr[i_star:i_star+1], kernel).view(N)  # K(x_j, x_i*)
    b = y[i_star] - torch.sum(a * y * K_train_sv)

    # scores on test: K(X_train, X_test)
    K_train_test = gram_matrix(x_tr, x_te, kernel)  # (N, M)
    scores = (a * y).view(1, -1) @ K_train_test     # (1, M)
    scores = scores.view(-1) + b
    return scores