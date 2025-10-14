# Introduction

## From-scratch GNNs for spatial transcriptomics.

Spatial transcriptomics captures gene expression while preserving tissue architecture, enabling the study of cellular organization and microenvironments. However, identifying coherent spatial domains - regions of similar expression patterns and spatial context — remains challenging.

Graph Neural Networks (GNNs) are great for this type of data because they can model both gene expression features and spatial neighborhood relationships. In this project, I implemented a mini Graph Autoencoder (GAE) from scratch in PyTorch to learn unsupervised spatial embeddings of tissue spots from a toy Visium H&E dataset provided by Squidpy.

The aim was to demonstrate how GNNs can capture spatially coherent patterns in gene expression and to compare these learned embeddings to traditional PCA-based clustering.

This project is inspired by [SpaGCN](https://github.com/jianhuupenn/SpaGCN) (*Hu et al.*), which extends this idea by integrating gene expression, spatial coordinates, and histology-derived features into a unified graph. While my implementation focuses on the core graph autoencoder concept, SpaGCN demonstrates how multimodal GNNs can reveal spatial domains and spatially variable genes in complex tissues.

A nice intro to deep learning on graphs and [GCNs by Jure Leskovec.](https://www.youtube.com/watch?v=MH4yvtgAR-4&list=PLoROMvodv4rPLKxIpqhjhPgdQy7imNkDn&index=19).


