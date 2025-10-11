# Introduction

From-scratch GNNs for Spatial Transcriptomics.

Spatial transcriptomics captures gene expression while preserving tissue architecture, enabling the study of cellular organization and microenvironments. However, identifying coherent spatial domains - regions of similar expression patterns and spatial context — remains challenging.

Graph Neural Networks (GNNs) are great for this type of data because they can model both gene expression features and spatial neighborhood relationships. In this project, I implemented a mini Graph Autoencoder (GAE) from scratch in PyTorch to learn unsupervised spatial embeddings of tissue spots from a toy Visium H&E dataset provided by Squidpy.

The aim was to demonstrate how GNNs can capture spatially coherent patterns in gene expression and to compare these learned embeddings to traditional PCA-based clustering.



