# Regression-based and Variational Autoencoder for DNA Methylation–Based Cell Type Estimation

## Introduction  

This project studies DNA fragments that circulate in the blood. These fragments come from many different cell types in the body.  
When tissues are damaged or diseased, they release more DNA than usual, so the mix of DNA in the blood changes.  
By figuring out which cell types the DNA comes from, we can get an early picture of tissue health.  

Examples:  
- In COVID-19 patients ([DOI: 10.1172/jci.insight.147610](https://doi.org/10.1172/jci.insight.147610))  
- In cancer ([DOI: 10.1038/nrc.2017.7](https://doi.org/10.1038/nrc.2017.7))  

This approach could become a powerful, non-invasive way to detect and monitor diseases using nothing more than a routine blood test.  

Traditionally, researchers study cell composition with methods like:  
- Microscopy and staining  
- Cell sorting  
- Single-cell RNA-sequencing  

Another way is to look at **DNA methylation**, which acts like a chemical fingerprint unique to each cell type ([overview](https://www.sciencedirect.com/topics/biochemistry-genetics-and-molecular-biology/methylome)).  
By reading these fingerprints, we can estimate how much DNA comes from each cell type in a mixed sample, such as blood.  

---

## Methods  

### Regression-based  

I applied regression-based deconvolution methods to estimate cell type proportions from bulk DNA methylation data.  
These approaches model the observed methylation profile as a weighted mixture of reference profiles from individual cell types.  

The tested methods were:  
- Non-negative least squares (NNLS) 
- Constrained optimization (sum-to-one)  
- Ridge regression  
- Lasso regression  
- Elastic net regression  

Each method was applied to simulated datasets with known ground-truth proportions.  
This allowed direct accuracy assessment under varying noise, numbers of CpGs, and cell type similarity.  

This design follows the benchmarking review by de Ridder et al.  
[DOI: 10.1038/s41467-024-48466-z](https://doi.org/10.1038/s41467-024-48466-z).  

---

### Deep learning – Variational Autoencoder  

In addition to regression, I implemented a **variational autoencoder (VAE)** with an output head for cell type proportions.  

- The VAE reconstructs CpG methylation profiles.  
- At the same time, it predicts the proportion of each cell type.  
- The encoder maps profiles into a latent space, from which both reconstructions and proportions are derived.  

Training used a combined loss function:  
1. Reconstruction error  
2. Supervised proportion error  
3. KL divergence regularization  

This allows the model to capture complex, non-linear patterns in methylation data that regression may miss.  

Performance was evaluated by comparing predicted proportions against the ground truth using **mean absolute error (MAE)**.  
The approach was inspired by *MethylNet* ([Levy et al., 2020](https://doi.org/10.1186/s12859-020-3443-8)).  

---

## Results  

- **Regression methods** (NNLS, ridge, lasso, elastic net) recovered cell type proportions.  
  - NNLS gave stable estimates.  
  - Ridge was robust to noise.  
  - Lasso and elastic net introduced sparsity by down-weighting minor cell types.  
  - MAE was generally low, though accuracy varied with cell type similarity and noise.  

- **VAE model** reconstructed methylation profiles and predicted proportions.  
  - Achieved test MAE of **0.21–0.22**.  
  - Captured overall trends but sometimes spread predictions across multiple cell types, even when truth was sparse.  
