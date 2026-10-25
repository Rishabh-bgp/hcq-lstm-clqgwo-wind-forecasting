"""HCQ-LSTM: LSTM cell whose gates are augmented with VQC sub-modules + physics-informed loss; CLQGWO tuner."""
import math, numpy as np, torch, torch.nn as nn
from .qlayer import vqc_layer

class HCQLSTMCell(nn.Module):
    def __init__(self, d_in, hidden, n_qubits=4, n_layers=1):
        super().__init__()
        self.cls = nn.Linear(d_in + hidden, 4 * hidden)
        self.pre = nn.Linear(d_in + hidden, n_qubits)
        self.vqc = nn.ModuleList([vqc_layer(n_qubits, n_layers) for _ in range(4)])  # one VQC per gate
        self.post = nn.ModuleList([nn.Linear(n_qubits, hidden) for _ in range(4)])
    def forward(self, x, h, c):
        z = torch.cat([x, h], -1); g = self.cls(z).chunk(4, -1)
        a = torch.tanh(self.pre(z)) * math.pi / 2
        g = [g[k] + self.post[k](self.vqc[k](a)) for k in range(4)]
        i, f, o, u = torch.sigmoid(g[0]), torch.sigmoid(g[1]), torch.sigmoid(g[2]), torch.tanh(g[3])
        c = f * c + i * u; return o * torch.tanh(c), c

class HCQLSTM(nn.Module):
    def __init__(self, n_feat=5, hidden=16, n_qubits=4, n_layers=1):
        super().__init__(); self.hidden = hidden
        self.cell = HCQLSTMCell(n_feat, hidden, n_qubits, n_layers); self.head = nn.Linear(hidden, 1)
    def forward(self, x):
        B = x.shape[0]; h = x.new_zeros(B, self.hidden); c = x.new_zeros(B, self.hidden)
        for t in range(x.shape[1]): h, c = self.cell(x[:, t], h, c)
        return self.head(h).squeeze(-1)

def physics_loss(lam_log=0.01, lam_mom=0.01, lam_spec=0.01):
    """Simplified physics penalties (in standardised units):
    log-law: forecasts should stay consistent with the last observed speed scaled by a log wall-layer ratio (single hub height => ratio 1);
    momentum: penalise unphysically large one-step accelerations;
    spectral: penalise deviation of the batch forecast-increment spectrum slope from Kolmogorov -5/3."""
    def f(model, X, p):
        last = X[:, -1, 0]
        l_log = torch.mean((p - last) ** 2) * 0.1
        l_mom = torch.mean(torch.relu(torch.abs(p - last) - 2.0) ** 2)
        s = torch.abs(torch.fft.rfft(p - p.mean())) ** 2
        k = torch.arange(1, s.shape[0], dtype=p.dtype)
        if len(k) > 3:
            lk, ls = torch.log(k), torch.log(s[1:] + 1e-8)
            slope = ((lk - lk.mean()) * (ls - ls.mean())).sum() / ((lk - lk.mean()) ** 2).sum()
            l_spec = (slope + 5 / 3) ** 2
        else: l_spec = p.new_zeros(())
        return lam_log * l_log + lam_mom * l_mom + lam_spec * l_spec
    return f

def tent_map(n, d, rng, mu=1.99):
    x = rng.random(d); out = []
    for _ in range(n):
        x = np.where(x < 0.5, mu * x, mu * (1 - x)); x = np.clip(x, 1e-6, 1 - 1e-6); out.append(x.copy())
    return np.array(out)

def levy(d, rng, beta=1.5):
    s = (math.gamma(1 + beta) * math.sin(math.pi * beta / 2) / (math.gamma((1 + beta) / 2) * beta * 2 ** ((beta - 1) / 2))) ** (1 / beta)
    return rng.normal(0, s, d) / np.abs(rng.normal(0, 1, d)) ** (1 / beta)

def clqgwo(obj, lb, ub, n_wolves=5, iters=4, seed=0, verbose=True):
    """Chaotic Levy-flight Quantum-inspired Grey Wolf Optimiser (minimisation over a box)."""
    rng = np.random.default_rng(seed); lb, ub = np.array(lb, float), np.array(ub, float); d = len(lb)
    X = lb + tent_map(n_wolves, d, rng) * (ub - lb)                      # chaotic initialisation
    theta = rng.uniform(0, np.pi / 2, (n_wolves, d))                     # qubit angles
    fit = np.array([obj(x) for x in X])
    for t in range(iters):
        order = np.argsort(fit); a_, b_, d_ = X[order[0]], X[order[1]], X[order[2]]
        a = 2 - 2 * t / max(iters - 1, 1)
        for i in range(n_wolves):
            new = np.zeros(d)
            for L in (a_, b_, d_):
                A = 2 * a * rng.random(d) - a; C = 2 * rng.random(d); new += L - A * np.abs(C * L - X[i])
            new /= 3
            theta[i] += 0.05 * np.pi * np.sign(a_ - X[i]) * rng.random(d)  # quantum rotation gate
            new += (np.cos(theta[i]) ** 2 - 0.5) * 0.1 * (ub - lb)
            new += 0.01 * levy(d, rng) * (new - a_)                          # Levy flight exploration
            new = np.clip(new, lb, ub); fn = obj(new)
            if fn < fit[i]: X[i], fit[i] = new, fn
        if verbose: print(f"CLQGWO iter {t+1}/{iters} best={fit.min():.4f}")
    return X[np.argmin(fit)], fit.min()
