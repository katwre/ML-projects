import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# =========================
# Simulate data
# =========================

# ---------- simulation helpers ----------
def simulate_hard(n_cpgs=200, n_celltypes=6, n_samples=4, noise_sd=0.07, seed=123):
    rng = np.random.default_rng(seed)
    # Start from a low-rank-ish base to induce collinearity between cell types
    base = rng.uniform(0.1, 0.9, size=(n_cpgs, 2))
    R = np.zeros((n_cpgs, n_celltypes))
    for k in range(n_celltypes):
        a, b = rng.uniform(0.2, 1.0, size=2)
        R[:, k] = np.clip(a*base[:, 0] + b*base[:, 1] + rng.normal(0, 0.05, n_cpgs), 0, 1)

    # True proportions, some samples deliberately sparse
    W_true = np.zeros((n_celltypes, n_samples))
    for j in range(n_samples):
        if j % 2 == 0:
            # sparse mix
            idx = rng.choice(n_celltypes, size=2, replace=False)
            w = np.zeros(n_celltypes); w[idx] = rng.uniform(0.05, 1.0, size=2); w /= w.sum()
        else:
            w = rng.dirichlet(np.ones(n_celltypes))
        W_true[:, j] = w

    # Mixtures + noise
    Y = np.clip(R @ W_true + rng.normal(0, 
                                        noise_sd, 
                                        size=(n_cpgs, n_samples)), 
                                        0, 1)

    # Simulate slight reference mismatch (batch effect / platform diff)
    R_ref = np.clip(R + rng.normal(0, 0.03, size=R.shape), 0, 1)

    cpgs = [f"cg{i:06d}" for i in range(1, n_cpgs+1)]
    celltypes = [f"CellType_{chr(65+i)}" for i in range(n_celltypes)]
    samples = [f"Sample_{i+1}" for i in range(n_samples)]

    ref_df = pd.DataFrame(R_ref, columns=celltypes, index=cpgs).reset_index().rename(columns={"index":"cpg"})
    mix_df = pd.DataFrame(Y, columns=samples, index=cpgs).reset_index().rename(columns={"index":"cpg"})
    truth  = pd.DataFrame(W_true, index=celltypes, columns=samples)
    return ref_df, mix_df, truth

def align_by_cpg(ref_df, mix_df):
    common = sorted(set(ref_df.cpg) & set(mix_df.cpg))
    R = ref_df.set_index("cpg").loc[common]
    Y = mix_df.set_index("cpg").loc[common]
    return R, Y
