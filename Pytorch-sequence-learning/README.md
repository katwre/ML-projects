# Sequence Learning using PyTorch

A hands-on PyTorch playground for learning neural network-based sequence modeling, using RNA sequences as examples.

An MLP (multi-layer perceptron) treats each position independently before pooling, so it mainly captures composition-level effects. A CNN applies convolutional filters across local windows and is good for motif detection. A Transformer encoder uses self-attention so each position can interact with all other positions, which makes it useful for modeling long-range dependencies. A VAE (variational autoencoder) learns a compressed latent representation of sequences and can be used for reconstruction and generation.

## What's covered:

- **MLP** - sequence-level regression with mean pooling; learns nucleotide composition but not order
- **CNN** - 1D convolutional filters that slide across the sequence to detect local motifs; global max pooling
- **Transformer encoder** - self-attention lets each position interact with all others, helping model long-range dependencies in sequences
- **Regularization** - dropout and batch normalization variants of the CNN
- **Training loop** - forward pass, MSE loss, backpropagation, Adam optimizer
- **Batched training** - stacking sequences and using DataLoader with Dataset
- **Regression / Classification** - use MSELoss for continuous targets or BCEWithLogitsLoss for binary prediction tasks
- **VAE** - encode sequences into a compact latent space, then decode them for reconstruction and synthetic sequence generation

## Architecture comparison

| Model | Captures | Key limitation |
|-------|----------|----------------|
| MLP + mean pool | Nucleotide composition | Ignores position/order |
| CNN | Local motifs (e.g. AUG) | Limited long-range context |
| Transformer | Long-range dependencies | More complex to train |
| VAE | Global latent structure; useful for reconstruction/generation | Not directly optimized for supervised prediction |

## Requirements

torch • numpy

