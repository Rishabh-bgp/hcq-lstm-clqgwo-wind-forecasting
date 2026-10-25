"""Train/evaluate HCQ-LSTM with physics-informed loss; optional CLQGWO tuning. Example: python train.py --epochs 2 --tune"""
import argparse, torch
from src.train_common import get_data, fit, evaluate
from src.hcq_lstm import HCQLSTM, physics_loss, clqgwo
ap = argparse.ArgumentParser()
ap.add_argument("--csv", default="data/synthetic/synthetic_scada.csv", help="SCADA-style CSV; pass \"\" to generate synthetic data in memory"); ap.add_argument("--n_synth", type=int, default=1000); ap.add_argument("--lookback", type=int, default=12)
ap.add_argument("--horizon", type=int, default=1); ap.add_argument("--epochs", type=int, default=5); ap.add_argument("--tune", action="store_true")
ap.add_argument("--wolves", type=int, default=3); ap.add_argument("--iters", type=int, default=2); ap.add_argument("--seed", type=int, default=42)
ap.add_argument("--quick", action="store_true", help="fast smoke run (<1 min): last 500 rows, few epochs")
a = ap.parse_args()
if a.quick: a.epochs = 1
a.max_rows = 500 if a.quick else None
torch.manual_seed(a.seed)
tr, va, te, sc = get_data(a.csv, a.n_synth, a.lookback, a.horizon, a.max_rows)
def build(x): return HCQLSTM(tr[0].shape[2], hidden=int(x[0])), dict(lr=float(10 ** x[1])), physics_loss(*[float(10 ** v) for v in x[2:5]])
x = [16, -2.5, -2, -2, -2]
if a.tune and a.quick: a.wolves, a.iters = 3, 1
if a.tune:
    def obj(x):
        torch.manual_seed(a.seed); m, kw, pl = build(x); return fit(m, tr, va, 1, extra_loss=pl, verbose=False, **kw)
    x, best = clqgwo(obj, [8, -3.5, -4, -4, -4], [24, -2, -1, -1, -1], a.wolves, a.iters, a.seed); print("CLQGWO best params", x.round(3))
torch.manual_seed(a.seed); m, kw, pl = build(x); fit(m, tr, va, a.epochs, extra_loss=pl, **kw); evaluate(m, te, sc, name="hcq_lstm")
