import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Variational autoencoder with proportion head
class VAEProp(nn.Module):
    '''
    This VAE is trained on CpG methylation profiles (X) to simultaneously
    reconstruct CpGs (unsupervised) and
    predict cell type proportions (T) (supervised).
    The final output is the predicted proportions per sample, which can be compared against the ground truth using MAE.

                ┌──────────────────────────────────────────────────────────┐
                │                          ENCODER                         │
    x (CpGs)    │                                                          │
    (n_cpgs)--> │  enc1: Linear(n_cpgs → h1) → ReLU                        │
                │        (2000 → 1024)                                     │
                │  enc2: Linear(h1 → h2) → ReLU                            │
                │        (1024 → 256)                                      │
                │                                                          │
                │  mu:   Linear(h2 → latent_dim)     (256 → 32)            │
                │  logv: Linear(h2 → latent_dim)     (256 → 32)            │
                └──────────────────────────────────────────────────────────┘
                                        │
                                        │   reparameterize:
                                        │   z = mu + eps * exp(0.5*logv)
                                        ▼
                            z (latent code, size = latent_dim = 32)
                            ├───────────────────────────────┐
                            │                               │
                            │                               │
            ┌───────────────▼──────────────┐   ┌───────────▼────────────────┐
            │          DECODER             │   │       PROPORTION HEAD       │
            │                              │   │                              │
            │  dec1: Linear(32 → 256)→ReLU │   │  head1: Linear(32 → H)→ReLU │
            │  dec2: Linear(256 → 1024)→ReLU│  │         (H = max(64, 4*k))  │
            │  out : Linear(1024 → 2000)   │   │  head2: Linear(H → k)        │
            │  sigmoid(out) → x_hat (2000) │   │  softmax → p_hat (k classes) │
            └──────────────────────────────┘   └──────────────────────────────┘
                         │                                   │
                         │                                   │
                 recon loss (MSE)                    proportion loss (MSE)
                         │                                   │
                         └──────────────┬────────────────────┘
                                        │
                                KL term from (mu, logv)
                                        │
                           total loss = recon + λ_prop * prop + λ_KL * KL
    '''
    def __init__(self, in_dim, latent_dim=32, hidden=(1024, 256), n_classes=6, kl_weight=1e-4, prop_weight=2.0):
        super().__init__()
        self.kl_weight = kl_weight
        self.prop_weight = prop_weight
        h1, h2 = hidden

        # Encoder
        self.enc1 = nn.Linear(in_dim, h1)
        self.enc2 = nn.Linear(h1, h2)
        self.mu   = nn.Linear(h2, latent_dim)
        self.logv = nn.Linear(h2, latent_dim)

        # Decoder
        self.dec1 = nn.Linear(latent_dim, h2)
        self.dec2 = nn.Linear(h2, h1)
        self.out  = nn.Linear(h1, in_dim)

        # Proportion head
        self.head1 = nn.Linear(latent_dim, max(64, n_classes*4))
        self.head2 = nn.Linear(max(64, n_classes*4), n_classes)

    def encode(self, x):
        h = F.relu(self.enc1(x))
        h = F.relu(self.enc2(h))
        return self.mu(h), self.logv(h)

    def reparameterize(self, mu, logv):
        std = torch.exp(0.5 * logv)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        h = F.relu(self.dec1(z))
        h = F.relu(self.dec2(h))
        return torch.sigmoid(self.out(h))

    def predict_prop(self, z):
        h = F.relu(self.head1(z))
        return F.softmax(self.head2(h), dim=-1)

    def forward(self, x):
        mu, logv = self.encode(x)
        z = self.reparameterize(mu, logv)
        x_hat = self.decode(z)
        p_hat = self.predict_prop(z)
        return x_hat, p_hat, mu, logv

    def loss_fn(self, x, x_hat, p_true, p_hat, mu, logv):
        recon = F.mse_loss(x_hat, x)
        sup   = F.mse_loss(p_hat, p_true)
        kl    = -0.5 * torch.mean(1 + logv - mu.pow(2) - logv.exp())
        return recon + self.prop_weight*sup + self.kl_weight*kl, recon, sup, kl



# Overfitting Sanity Check for VAEProportions
def overfit_sanity(VAEClass, X_train, T_train, X_test, T_test,
                   in_dim, n_classes,
                   latent_dim=32, hidden=(1024,256),
                   kl_weight=0.0,      # turn off KL to make overfitting easier
                   prop_weight=3.0,    # emphasize proportion head
                   lr=1e-3, weight_decay=0.0,
                   n_tiny=16, epochs=300, batch=16, verbose_every=50):

    # pick a tiny subset
    n_tiny = min(n_tiny, len(X_train))
    idx = torch.randperm(len(X_train))[:n_tiny]
    Xtiny = X_train[idx].to(DEVICE)
    Ttiny = T_train[idx].to(DEVICE)

    # fresh model & opt
    model = VAEClass(in_dim=in_dim, latent_dim=latent_dim, hidden=hidden,
                     n_classes=n_classes, kl_weight=kl_weight, prop_weight=prop_weight).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    # histories (per-epoch)
    hist_tr = {"loss":[], "recon":[], "sup":[], "kl":[], "mae":[]}
    hist_te = {"loss":[], "recon":[], "sup":[], "kl":[], "mae":[]}

    # --- training over tiny set ---
    for ep in range(1, epochs+1):
        model.train()
        perm = torch.randperm(n_tiny, device=DEVICE)
        tot = {"loss":0.0, "recon":0.0, "sup":0.0, "kl":0.0}

        for i in range(0, n_tiny, batch):
            j = perm[i:i+batch]
            xb, tb = Xtiny[j], Ttiny[j]
            opt.zero_grad()
            x_hat, p_hat, mu, logv = model(xb)
            loss, recon, sup, kl = model.loss_fn(xb, x_hat, tb, p_hat, mu, logv)
            loss.backward(); opt.step()
            n = xb.size(0)
            tot["loss"]  += loss.item()*n
            tot["recon"] += recon.item()*n
            tot["sup"]   += sup.item()*n
            tot["kl"]    += kl.item()*n

        # averages on tiny-train
        for k in tot: tot[k] /= n_tiny
        with torch.no_grad():
            _, p_hat_tr, _, _ = model(Xtiny)
            mae_tr = torch.mean(torch.abs(p_hat_tr - Ttiny)).item()
        for k in ["loss","recon","sup","kl"]: hist_tr[k].append(tot[k])
        hist_tr["mae"].append(mae_tr)

        # evaluate on full test each epoch
        model.eval()
        with torch.no_grad():
            totals_te = {"loss":0.0, "recon":0.0, "sup":0.0, "kl":0.0}
            Nte = len(X_test)
            for i in range(0, Nte, 256):
                xb = X_test[i:i+256].to(DEVICE)
                tb = T_test[i:i+256].to(DEVICE)
                x_hat, p_hat, mu, logv = model(xb)
                loss, recon, sup, kl = model.loss_fn(xb, x_hat, tb, p_hat, mu, logv)
                n = xb.size(0)
                totals_te["loss"]  += loss.item()*n
                totals_te["recon"] += recon.item()*n
                totals_te["sup"]   += sup.item()*n
                totals_te["kl"]    += kl.item()*n
            for k in totals_te: totals_te[k] /= Nte
            _, p_hat_te, _, _ = model(X_test.to(DEVICE))
            mae_te = torch.mean(torch.abs(p_hat_te - T_test.to(DEVICE))).item()

        for k in ["loss","recon","sup","kl"]: hist_te[k].append(totals_te[k])
        hist_te["mae"].append(mae_te)

        if ep % verbose_every == 0 or ep == 1:
            print(f"[TinyFit] Epoch {ep:03d} | "
                  f"train: loss={tot['loss']:.4f} recon={tot['recon']:.4f} prop={tot['sup']:.4f} kl={tot['kl']:.6f} mae={mae_tr:.4f} | "
                  f"test:  loss={totals_te['loss']:.4f} recon={totals_te['recon']:.4f} prop={totals_te['sup']:.4f} kl={totals_te['kl']:.6f} mae={mae_te:.4f}")

    # final one-line summary like before
    print(f"\nTiny-set MAE (should be small): {hist_tr['mae'][-1]:.4f}")
    print(f"Test-set MAE (likely larger):   {hist_te['mae'][-1]:.4f}")

    # --- plots: train vs test (loss components + MAE) ---
    x = range(1, epochs+1)

    plt.figure(figsize=(10,6))
    plt.plot(x, hist_tr["loss"],  label="train total")
    plt.plot(x, hist_te["loss"],  "--", label="test total")
    plt.plot(x, hist_tr["recon"], label="train recon")
    plt.plot(x, hist_te["recon"], "--", label="test recon")
    plt.plot(x, hist_tr["sup"],   label="train prop")
    plt.plot(x, hist_te["sup"],   "--", label="test prop")
    plt.plot(x, hist_tr["kl"],    label="train KL")
    plt.plot(x, hist_te["kl"],    "--", label="test KL")
    plt.xlabel("Epoch"); plt.ylabel("Loss")
    plt.title("Overfit sanity: losses (train tiny vs test)")
    plt.legend(ncol=2); plt.tight_layout(); plt.show()

    plt.figure(figsize=(8,5))
    plt.plot(x, hist_tr["mae"], label="train MAE")
    plt.plot(x, hist_te["mae"], "--", label="test MAE")
    plt.xlabel("Epoch"); plt.ylabel("Proportion MAE")
    plt.title("Overfit sanity: MAE (train tiny vs test)")
    plt.legend(); plt.tight_layout(); plt.show()

    return model
