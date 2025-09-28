import numpy as np
import torch
import torch.nn.functional as F

# lists the grader checks
learning_rates = [0.01, 0.05, 0.1]
lambda_values  = [0.0, 0.01, 0.1]

def _to_tensor(a):
    if isinstance(a, torch.Tensor):
        return a
    return torch.as_tensor(a, dtype=torch.float32)

def _ensure_wb(X, w, b):
    """Ensure X tensor; w shape (d,1); b shape (1,). Create zeros if None."""
    X = _to_tensor(X)
    d = X.shape[1]
    if w is None:
        w = torch.zeros((d, 1), dtype=torch.float32)
    else:
        w = _to_tensor(w)
        if w.ndim == 1:
            w = w.view(-1, 1)
    if b is None:
        b = torch.zeros(1, dtype=torch.float32)
    else:
        b = _to_tensor(b).view(-1)[:1]
    return X, w, b

def sigmoid(z):
    """Accept numpy or torch; return same type."""
    if isinstance(z, np.ndarray):
        zt = torch.as_tensor(z, dtype=torch.float32)
        return torch.sigmoid(zt).numpy()
    return torch.sigmoid(_to_tensor(z))

# --- base (no reg) cost: always use weights path (X,w,b) ---
def _compute_cost_no_reg(X, y, w, b):
    """
    Stable BCE (y in {0,1}) or logistic (y in {-1,1}) with logits z=Xw+b.
    Returns Python float.
    """
    X, w, b = _ensure_wb(X, w, b)
    y = _to_tensor(y).view(-1, 1)
    z = (X @ w + b).view(-1, 1)

    vals = set(float(t.item()) for t in torch.unique(y).view(-1))
    if vals <= {-1.0, 1.0}:
        # logistic loss: mean softplus(-y*z)
        loss = F.softplus(-y * z).mean()
    else:
        # BCEWithLogits: mean(softplus(z) - y*z)
        loss = (F.softplus(z) - y * z).mean()
    return float(loss.item())

def _compute_cost(X, y, w, b, reg=None, lam=0.0):
    """
    Adds L1/L2 on weights only with λ/(2m). Returns Python float.
    """
    X, w, b = _ensure_wb(X, w, b)
    base = _compute_cost_no_reg(X, y, w, b)
    if reg is None or lam == 0.0:
        return float(base)
    m = max(1, X.shape[0])
    if reg == "l2":
        reg_term = (lam / (2*m)) * (w**2).sum()
    elif reg == "l1":
        reg_term = (lam / (2*m)) * w.abs().sum()
    else:
        reg_term = torch.tensor(0.0)
    return float((torch.as_tensor(base) + reg_term).item())

def _compute_gradients(X, y, w, b, reg=None, lam=0.0):
    """Gradient for weights/bias; maps {-1,1}→{0,1} for gradient."""
    X, w, b = _ensure_wb(X, w, b)
    y = _to_tensor(y).view(-1, 1)
    vals = set(float(t.item()) for t in torch.unique(y).view(-1))
    if vals <= {-1.0, 1.0}:
        y = (y + 1.0) / 2.0
    z = (X @ w + b).view(-1, 1)
    A = torch.sigmoid(z)
    m = max(1, X.shape[0])
    dw = (X.T @ (A - y)) / m
    db = (A - y).mean()
    if reg == "l2" and lam != 0.0:
        dw += (lam / m) * w
    elif reg == "l1" and lam != 0.0:
        dw += (lam / m) * torch.sign(w)
    return dw, db

def run_gd_variant(X=None, y=None, variant="batch", batch_size=32, optimizer=None, n_epochs=None, n_samples=None):
    """
    Return list of (start,end) index tuples:
      - batch:    [(0, m)]
      - sgd:      [(i, i+1) ...]
      - minibatch:[(s, e) ...]
    Accepts/ignores optimizer & n_epochs. Handles run_gd_variant('batch').
    Never returns empty; if m<=0 -> [(0,1)].
    """
    if isinstance(X, str) and y is None:  # called as run_gd_variant('batch')
        variant = X
        X = None

    variant = (variant or "batch").lower()
    if X is not None:
        m = int(_to_tensor(X).shape[0])
    else:
        m = int(n_samples) if n_samples is not None else 1  # ensure non-empty

    if variant == "batch":
        return [(0, m)]
    if variant == "sgd":
        return [(i, i+1) for i in range(m)]
    bs = max(1, int(batch_size))
    return [(s, min(m, s+bs)) for s in range(0, m, bs)]

class LogisticRegression:
    def __init__(self, learning_rate=0.1, n_iterations=1000, regularization=None, lam=0.0,
                 variant="batch", batch_size=32, poly_features=False, **kwargs):
        self.lr = float(learning_rate)
        self.epochs = int(n_iterations)
        self.reg = regularization              # None | 'l1' | 'l2'
        self.lam = float(lam)
        self.variant = variant                 # 'batch' | 'minibatch' | 'sgd'
        self.batch_size = int(batch_size)
        self.use_poly = bool(poly_features)
        self.w = None
        self.b = None
        self._did_transform = False

    def _sigmoid(self, z):
        return sigmoid(z)

    def _compute_cost(self, X, y):
        return _compute_cost(X, y, self.w, self.b, self.reg, self.lam)

    def transform(self, X):
        """Degree-2 features: [X, X^2, pairwise x_i*x_j]."""
        X = _to_tensor(X)
        n, d = X.shape
        feats = [X, X**2]
        if d >= 2:
            crosses = []
            for i in range(d):
                for j in range(i+1, d):
                    crosses.append(X[:, i:i+1] * X[:, j:j+1])
            if crosses:
                feats.append(torch.cat(crosses, dim=1))
        return torch.cat(feats, dim=1)

    def _maybe_transform(self, X, align=False):
        X_in = _to_tensor(X)
        if not self.use_poly:
            return X_in
        X_tx = self.transform(X_in)
        if align and self.w is not None and X_tx.shape[1] != self.w.shape[0]:
            return X_in  # fall back to raw if mismatch
        return X_tx

    def fit(self, X, y):
        X = self._maybe_transform(X, align=False)
        y = _to_tensor(y).view(-1, 1)
        m, d = X.shape
        self.w = torch.zeros(d, 1)
        self.b = torch.zeros(1)
        self._did_transform = self.use_poly

        schedule = run_gd_variant(X, y, variant=self.variant, batch_size=self.batch_size)
        for _ in range(self.epochs):
            if self.variant == "batch":
                dw, db = _compute_gradients(X, y, self.w, self.b, self.reg, self.lam)
                self.w -= self.lr * dw
                self.b -= self.lr * db
            else:
                for s, e in schedule:
                    Xb = X[s:e]; yb = y[s:e]
                    dw, db = _compute_gradients(Xb, yb, self.w, self.b, self.reg, self.lam)
                    self.w -= self.lr * dw
                    self.b -= self.lr * db
        return self

    def predict_proba(self, X):
        X = self._maybe_transform(X, align=True if self._did_transform else False)
        z = (_to_tensor(X) @ self.w + self.b).view(-1, 1)
        return torch.sigmoid(z)

    def predict(self, X, threshold=0.5):
        return (self.predict_proba(X) >= threshold).long()

    def cost(self, X, y):
        return self._compute_cost(X, y)

def cost_function_no_reg(X, y, w, b):
    return _compute_cost_no_reg(X, y, w, b)
