# =========================
# REGRESSION DECONVOLUTION
# =========================

import numpy as np
import pandas as pd
from scipy.optimize import nnls, minimize
from sklearn.linear_model import Lasso, ElasticNet

# ---------- solvers ----------
def solve_nnls(R, y, sum_to_one=True):
    w, _ = nnls(R, y)
    if sum_to_one and w.sum() > 0:
        w = w / w.sum()
    return w

def solve_nnls_sum1(R, y):
    n = R.shape[1]
    bounds = [(0, None)] * n
    cons = ({"type": "eq", "fun": lambda w: np.sum(w) - 1.0},)
    def obj(w):
        r = R @ w - y
        return float(r @ r)
    res = minimize(obj, x0=np.ones(n)/n, bounds=bounds, constraints=cons, method="SLSQP")
    return res.x if res.success else solve_nnls(R, y, sum_to_one=True)

def solve_ridge_nonneg(R, y, alpha=1.0, sum_to_one=True):
    n = R.shape[1]
    bounds = [(0, None)] * n
    cons = ({"type":"eq", "fun": lambda w: np.sum(w) - 1.0},) if sum_to_one else ()
    def obj(w):
        r = R @ w - y
        return float(r @ r + alpha * (w @ w))
    res = minimize(obj, x0=np.ones(n)/n, bounds=bounds, constraints=cons, method="SLSQP")
    return res.x if res.success else solve_nnls(R, y, sum_to_one=sum_to_one)

def solve_lasso_pos(R, y, alpha=0.05, sum_to_one=True):
    model = Lasso(alpha=alpha, positive=True, fit_intercept=False, max_iter=10000)
    model.fit(R, y)
    w = model.coef_
    if sum_to_one and w.sum() > 0:
        w = w / w.sum()
    return w

def solve_elastic_pos(R, y, alpha=0.08, l1_ratio=0.7, sum_to_one=True):
    model = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, positive=True, fit_intercept=False, max_iter=20000)
    model.fit(R, y)
    w = model.coef_
    if sum_to_one and w.sum() > 0:
        w = w / w.sum()
    return w

def deconvolve_all(ref_aln, mix_aln, sum_to_one=True):
    R = ref_aln.values
    celltypes = list(ref_aln.columns)
    samples = list(mix_aln.columns)
    out = {m: pd.DataFrame(0.0, index=celltypes, columns=samples)
           for m in ["nnls","nnls_sum1","ridge","lasso","elastic_net"]}
    for s in samples:
        y = mix_aln[s].values
        out["nnls"][s]        = solve_nnls(R, y, sum_to_one=sum_to_one)
        out["nnls_sum1"][s]   = solve_nnls_sum1(R, y)
        out["ridge"][s]       = solve_ridge_nonneg(R, y, alpha=1.0, sum_to_one=sum_to_one)
        out["lasso"][s]       = solve_lasso_pos(R, y, alpha=0.05, sum_to_one=sum_to_one)
        out["elastic_net"][s] = solve_elastic_pos(R, y, alpha=0.08, l1_ratio=0.7, sum_to_one=sum_to_one)
    return out
