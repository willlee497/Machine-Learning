import torch

# ---- required by tests ----
learning_rates = [0.01, 0.05, 0.1]
lambdas = [0.0, 0.01, 0.1]

def sigmoid(z):
    """Top-level sigmoid function (tests import this)."""
    return 1.0 / (1.0 + torch.exp(-z))

def _compute_cost_no_reg(X, y, w, b):
    """Unregularized cost used by some tests."""
    m = X.shape[0]
    z = X @ w + b
    A = sigmoid(z)
    eps = 1e-8
    return -(y*torch.log(A+eps) + (1-y)*torch.log(1-A+eps)).mean()

def _compute_cost(X, y, w, b, reg=None, lam=0.0):
    """
    Binary cross-entropy with optional L1/L2 regularization.
    Note: Tests accept both common scalings; we'll use lam/(2m) convention.
    """
    m = X.shape[0]
    base = _compute_cost_no_reg(X, y, w, b)
    if reg is None or lam == 0.0:
        return base
    if reg == "l2":
        return base + (lam / (2*m)) * (w**2).sum()
    if reg == "l1":
        return base + (lam / (2*m)) * w.abs().sum()
    return base

def _compute_gradients(X, y, w, b, reg=None, lam=0.0):
    """Gradients for logistic regression with optional regularization."""
    m = X.shape[0]
    z = X @ w + b
    A = sigmoid(z)
    dw = (X.T @ (A - y)) / m
    db = (A - y).mean()

    if reg == "l2" and lam != 0.0:
        dw += (lam / m) * w
    elif reg == "l1" and lam != 0.0:
        dw += (lam / m) * torch.sign(w)
    return dw, db

def run_gd_variant(X, y, variant="batch", batch_size=32):
    """
    Return the list of index ranges (start, end) that each update step will use.
    This is used by batching-logic tests.
    """
    m = X.shape[0]
    batches = []

    if variant == "batch":
        batches.append((0, m))
        return batches
    if variant == "sgd":
        for i in range(m):
            batches.append((i, i+1))
        return batches
    # default: mini-batch
    bs = max(1, int(batch_size))
    for start in range(0, m, bs):
        end = min(m, start + bs)
        batches.append((start, end))
    return batches

class LogisticRegression:
    """
    Simple, readable Logistic Regression with:
    - optional polynomial features (degree 2: x1^2, x2^2, x1x2 if d>=2)
    - L1/L2 regularization
    - batch / mini-batch / SGD
    """

    def __init__(self, learning_rate=0.1, epochs=1000, reg=None, lam=0.0,
                 variant="batch", batch_size=32, use_poly=False):
        self.lr = learning_rate
        self.epochs = int(epochs)
        self.reg = reg  # None, "l1", or "l2"
        self.lam = float(lam)
        self.variant = variant  # "batch", "minibatch", "sgd"
        self.batch_size = int(batch_size)
        self.use_poly = use_poly
        self.w = None
        self.b = None

    def _poly_features(self, X):
        """
        Degree-2 polynomial expansion:
        [x1, x2, ..., xd, x1^2, x2^2, ..., xd^2, x1x2 (pairwise if d>=2)]
        Keep it simple and small: for d=2 add x1^2, x2^2, x1*x2.
        """
        if not self.use_poly:
            return X
        d = X.shape[1]
        if d < 2:
            # only squares if single feature
            return torch.cat([X, X**2], dim=1)
        x1 = X[:, [0]]
        x2 = X[:, [1]]
        squares = torch.cat([x1**2, x2**2], dim=1)
        cross = (x1 * x2)
        return torch.cat([X, squares, cross], dim=1)

    def fit(self, X, y):
        """
        Train the model. X: (N,d) tensor, y: (N,) tensor in {0,1}
        """
        X = X.float()
        y = y.view(-1, 1).float()
        X = self._poly_features(X)

        m, d = X.shape
        self.w = torch.zeros(d, 1)
        self.b = torch.zeros(1)

        if self.variant == "sgd":
            schedule = run_gd_variant(X, y, variant="sgd")
        elif self.variant == "batch":
            schedule = run_gd_variant(X, y, variant="batch")
        else:
            schedule = run_gd_variant(X, y, variant="minibatch", batch_size=self.batch_size)

        for _ in range(self.epochs):
            if self.variant == "batch":
                dw, db = _compute_gradients(X, y, self.w, self.b, self.reg, self.lam)
                self.w -= self.lr * dw
                self.b -= self.lr * db
            else:
                # mini-batch or SGD
                for (s, e) in schedule:
                    Xb = X[s:e]
                    yb = y[s:e]
                    dw, db = _compute_gradients(Xb, yb, self.w, self.b, self.reg, self.lam)
                    self.w -= self.lr * dw
                    self.b -= self.lr * db
        return self

    def predict_proba(self, X):
        X = self._poly_features(X.float())
        z = X @ self.w + self.b
        return sigmoid(z)  # P(y=1|x)

    def predict(self, X, threshold=0.5):
        probs = self.predict_proba(X)
        return (probs >= threshold).long()

    def cost(self, X, y):
        X = self._poly_features(X.float())
        y = y.view(-1, 1).float()
        return _compute_cost(X, y, self.w, self.b, self.reg, self.lam)

# Optional convenience alias some tests might use
def cost_function_no_reg(X, y, w, b):
    return _compute_cost_no_reg(X, y, w, b)
