import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# -------------------------
# Simulate input dataset
# -------------------------

# -------------------------
# Toy data simulator
# Cell types are generated with smoother, more distinct sinusoidal patterns.
# Less correlation between columns of R.
# The anchor Rref matches the true R (no mismatch).
# This is a much easier toy problem, so the Transformer quickly learns neat diagonal scatterplots and high correlations
# -------------------------

def simulate_data(P=1000, K=5, N=500, noise_sigma=0.02):
    # reference profiles R in [0,1], regions x celltypes
    # structure: each cell type has slightly different baseline + smooth bumps
    x = np.linspace(0, 1, P, dtype=np.float32)
    R = []
    rng = np.random.default_rng(123)
    for k in range(K):
        base = 0.3 + 0.4 * np.sin(2*np.pi*(k+1)*x)  # smooth pattern per cell
        base = base + 0.05 * rng.normal(size=P)     # small jitter
        base = np.clip(base, 0, 1)
        R.append(base)
    R = np.stack(R, axis=1).astype(np.float32)      # [P,K]

    # sample proportions H on simplex via Dirichlet
    H = rng.dirichlet(alpha=np.ones(K, dtype=np.float32), size=N).astype(np.float32)  # [N,K]
    # mix: Y = H @ R^T  -> [N,P]
    Y = (H @ R.T).astype(np.float32)
    if noise_sigma > 0:
        Y += noise_sigma * rng.normal(size=Y.shape).astype(np.float32)
    Y = np.clip(Y, 0, 1)
    # (reference, mixtures, true proportions)
    return R, Y, H


# -------------------------
# Dataset
# -------------------------
class SimpleDataset(torch.utils.data.Dataset):
    def __init__(self, reg_ids, reg_feats, Y, Htrue):
        self.reg_ids = torch.tensor(reg_ids, dtype=torch.long) # region indices, indices for embedding lookup
        self.reg_feats = torch.tensor(reg_feats, dtype=torch.float32) # region features, methylation value
        self.Y = torch.tensor(Y, dtype=torch.float32)
        self.Htrue = torch.tensor(Htrue, dtype=torch.float32) # Ground truth proportion
    def __len__(self):
        # get number of samples
        return self.Y.shape[0]
    def __getitem__(self, i):
        # retrieve single samples
        return self.reg_ids[i], self.reg_feats[i], self.Y[i], self.Htrue[i]

# -------------------------
# Tiny Deconv-Transformer
#   - region id embedding + feature projection
#   - TransformerEncoder over tokens (regions)
#   - mean pool -> MLP head -> softmax proportions
#   - learns W (regions x celltypes), anchored to R
#
# Self-attention = compare regions.
# Heads learn to notice co-varying bumps/troughs across regions that are characteristic of each cell type’s sinusoid in R
# Think: “these peaks rise and fall together → smells like cell type 3.”
#
# Encoder output = a summary of the pattern.
# After mixing information across regions, you have a sequence of contextualized vectors that encode the global shape of the sample’s signal.
#
# Head = turn shape → proportions.
# prop_head + W compress that sequence to K numbers and (likely with a softmax) put them on the simplex. Passing R_ref=R_np gives the model an anchor so the mapping aligns with your true cell-type axes instead of arbitrary rotations.
#
# nice intro about encoder transformers by statquest: https://www.youtube.com/watch?v=GDN649X_acE
# -------------------------
class TinyDeconvTransformer(nn.Module):
    def __init__(self, n_regions, feat_dim, d_model, n_layers, n_heads, n_cells, R_ref, dropout=0.1):
        super().__init__()
        self.n_regions = n_regions # number of unique regions (tokens)
        self.n_cells = n_cells # number of cell types (output dim)

        self.reg_emb = nn.Embedding(n_regions, d_model) # Maps region IDs to dense vectors of size d_model
        self.feat_proj = nn.Linear(feat_dim, d_model) # Projects input features to the same dimension as embeddings.

        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model, #  dimensionality of the input embeddings, size of the vectors
            nhead=n_heads, # number of attention heads in the multi-head self-attention
            dim_feedforward=4*d_model, # size of the feedforward network hidden layer inside the Transformer layer
            dropout=dropout, # regularization
            batch_first=True,  #  input tensors have the shape [batch_size, seq_len, d_model]
            activation="gelu", # activation function used in the feedforward network, GELU (Gaussian Error Linear Unit) is commonly used in Transformers for smoother gradients
            # Layer Normalization is applied before the self-attention and feedforward sublayers
            # This is known as Pre-Norm and can improve training stability
            norm_first=True
        )
        # 1. Multi-head attention
        #Each token (region) produces:
        # a Query vector → “what am I looking for?”
        # a Key vector → “how should others find me?”
        # a Value vector → “what information do I offer?”

        # 2. Feedforward Network (FFN)
        # After attention, each token goes through the same two-layer MLP:
        # FFN(x) = Linear(x, 4d) --(GELU)--> Linear(4d, d)
        # This gives the model nonlinear capacity to reshape features per token.

        # 3. Dropout + Residuals + Normalization
        
        # Processes the token representations using self-attention and feedforward layers.
        self.encoder = nn.TransformerEncoder(enc_layer, 
                                             num_layers=n_layers) # stack of identical encoder layers

        # MLP head to produce logits for each cell type
        self.prop_head = nn.Sequential(
            # A fully connected layer that transforms the input
            # It ensures that the input embeddings are mapped to the correct dimensionality for the next layer in the prop_head.
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, n_cells)
        )

        # learnable W initialized near R_ref
        R_t = torch.tensor(R_ref, dtype=torch.float32)
        self.W = nn.Parameter(R_t.clone())  # [P,K]

        # init
        nn.init.trunc_normal_(self.reg_emb.weight, std=0.02)
        nn.init.trunc_normal_(self.feat_proj.weight, std=0.02)
        nn.init.zeros_(self.feat_proj.bias)
        for m in self.prop_head:
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)


    def forward(self, reg_ids, reg_feats):
        # reg_ids: [B, P?] indices 0..n_regions-1
        # reg_feats: [B, P?, feat_dim] e.g., beta signal (and possibly other features)
        #  Converts region IDs to dense vectors.
        e = self.reg_emb(reg_ids)                   # [B,P?,D]
        # Projects input features to the same dimension as embeddings
        f = self.feat_proj(reg_feats)               # [B,P?,D]
        # Combines embeddings and projected features.
        x = e + f                                   # token rep
        #  Processes the token representations.
        x = self.encoder(x)                         # [B,P?,D]
        # Aggregates the sequence output into a single vector per sample.
        x = x.mean(dim=1)                           # [B,D] (mean pool)
        logits = self.prop_head(x)                  # [B,K]
        # Converts logits to proportions
        H_hat = torch.softmax(logits, dim=-1)       # 
        return H_hat


# -------------------------
# Train and eval
# -------------------------

# Training and Evaluation Functions
def train_transformer(model, train_loader, val_loader, R_ref, device, params):
    """
    Train the Transformer model.

    Args:
        model: The Transformer model to train.
        train_loader: DataLoader for training data.
        val_loader: DataLoader for validation data.
        R_ref: Reference matrix for anchoring.
        device: Device to run the training on (e.g., 'cuda' or 'cpu').
        params: Dictionary of training parameters (e.g., epochs, learning rate).

    Returns:
        train_history: List of training losses per epoch.
        val_history: List of validation losses per epoch.
        r_history: List of mean Pearson correlations per epoch.
        mae_history: List of mean MAE per epoch.
    """
    epochs = params['epochs']
    lambda_anchor = params['lambda_anchor']
    optimizer = params['optimizer']
    P_SUB = params['P_SUB']

    R_t = torch.tensor(R_ref, dtype=torch.float32, device=device)

    
# Training and Evaluation Functions
def train_transformer(model, train_loader, val_loader, R_ref, device, params):
    """
    Train the Transformer model.

    Args:
        model: The Transformer model to train.
        train_loader: DataLoader for training data.
        val_loader: DataLoader for validation data.
        R_ref: Reference matrix for anchoring.
        device: Device to run the training on (e.g., 'cuda' or 'cpu').
        params: Dictionary of training parameters (e.g., epochs, learning rate).

    Returns:
        train_history: List of training losses per epoch.
        val_history: List of validation losses per epoch.
        r_history: List of mean Pearson correlations per epoch.
        mae_history: List of mean MAE per epoch.
    """
    epochs = params['epochs']
    lambda_anchor = params['lambda_anchor']
    optimizer = params['optimizer']
    P_SUB = params['P_SUB']

    R_t = torch.tensor(R_ref, dtype=torch.float32, device=device)

    # histories for plotting
    train_history, val_history, r_history, mae_history = [], [], [], []


    def subsample_regions(reg_ids, reg_feats, Y, P_SUB):
        """Helper function to subsample regions."""
        sub_idx = torch.randperm(R_ref.shape[0], device=device)[:P_SUB]
        reg_ids_sub = reg_ids[:, sub_idx]
        reg_feats_sub = reg_feats[:, sub_idx, :]
        Y_sub = Y[:, sub_idx]
        W_sub = model.W[sub_idx, :]
        R_sub = R_t[sub_idx, :]
        return reg_ids_sub, reg_feats_sub, Y_sub, W_sub, R_sub

    # ---- train ----
    for epoch in range(1, epochs+1):
    # -------- train --------
        model.train(); tr_loss = 0.0
        for reg_ids_b, reg_feats_b, Y_b, _ in train_loader:
            reg_ids_b   = reg_ids_b.to(device)        # [B,P]
            reg_feats_b = reg_feats_b.to(device)      # [B,P,1]
            Y_b         = Y_b.to(device)              # [B,P]

            # Subsample regions
            reg_ids_sub, reg_feats_sub, Y_sub, W_sub, R_sub = subsample_regions(
                reg_ids_b, reg_feats_b, Y_b, P_SUB
            )
            # Forward pass
            H_hat = model(reg_ids_sub, reg_feats_sub)
            Y_hat = (W_sub @ H_hat.T).T

            # Compute losses
            mix_loss = F.mse_loss(Y_hat, Y_sub)
            anchor_loss = F.mse_loss(W_sub, R_sub)
            loss = mix_loss + lambda_anchor * anchor_loss

            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            tr_loss += loss.item() * Y_b.size(0)
        tr_loss /= len(train_loader.dataset)

        # -------- validate --------
        model.eval(); va_loss = 0.0; 
        all_Hhat = []; all_Htrue = []
        with torch.no_grad():
            for reg_ids_b, reg_feats_b, Y_b, Htrue_b in val_loader:
                reg_ids_b   = reg_ids_b.to(device)
                reg_feats_b = reg_feats_b.to(device)
                Y_b         = Y_b.to(device)
                Htrue_b     = Htrue_b.to(device)

                # Subsample regions
                reg_ids_sub, reg_feats_sub, Y_sub, W_sub, R_sub = subsample_regions(
                    reg_ids_b, reg_feats_b, Y_b, P_SUB
                )

                # Forward pass
                H_hat = model(reg_ids_sub, reg_feats_sub)
                Y_hat = (W_sub @ H_hat.T).T

                # Compute losses
                mix_loss = F.mse_loss(Y_hat, Y_sub)
                anchor_loss = F.mse_loss(W_sub, R_sub)
                loss = mix_loss + lambda_anchor * anchor_loss

                va_loss += loss.item() * Y_b.size(0)
                all_Hhat.append(H_hat.detach().cpu().numpy())
                all_Htrue.append(Htrue_b.detach().cpu().numpy())

        va_loss  /= len(val_loader.dataset)
        all_Hhat  = np.concatenate(all_Hhat, axis=0)
        all_Htrue = np.concatenate(all_Htrue, axis=0)
        mean_r    = pearson_colwise_mean(all_Hhat, all_Htrue)
        mean_mae  = mae_colwise_mean(all_Hhat, all_Htrue)

        # Append metrics to history
        train_history.append(tr_loss)
        val_history.append(va_loss)
        r_history.append(float(mean_r))
        mae_history.append(float(mean_mae))

        print(f"Epoch {epoch:02d} | train {tr_loss:.4f} | val {va_loss:.4f} | mean r={mean_r:.3f} | mean MAE={mean_mae:.3f}")


    return train_history, val_history, r_history, mae_history, all_Hhat, all_Htrue



# -------------------------
# Metrics
# -------------------------
def pearson_colwise_mean(Hhat_np, Htrue_np):
    '''
    Compute mean Pearson correlation coefficient across columns (cell types).
    Hhat_np: [N, K] numpy array of estimated proportions
    Htrue_np: [N, K] numpy array of true proportions
    Returns: mean Pearson r across K cell types'''
    K = Htrue_np.shape[1]
    rs = []
    for k in range(K):
        a = Htrue_np[:, k]; b = Hhat_np[:, k]
        if np.allclose(a.std(), 0) or np.allclose(b.std(), 0):
            rs.append(0.0)
        else:
            r = np.corrcoef(a, b)[0, 1]
            rs.append(float(r))
    return float(np.nanmean(rs))

def mae_colwise_mean(Hhat_np, Htrue_np):
    '''
    Computes the Mean Absolute Error (MAE) for each output dimension.
    Returns the mean MAE across all dimensions.
    '''
    return float(np.mean(np.abs(Hhat_np - Htrue_np)))

