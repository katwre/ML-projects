# Introduction to State-Space Models (SSMs) with PyMC

This notebook provides a **gentle introduction** to **State-Space Models (SSMs)** using [PyMC](https://www.pymc.io/), [ArviZ](https://python.arviz.org/), and [Matplotlib](https://matplotlib.org/).  
The goal is to demonstrate, step by step, how to:

- Define a **simple linear Gaussian state-space model**
- Perform **prior predictive checks**
- Fit the model to simulated data using **Bayesian inference**
- Evaluate the **posterior predictive fit**
- Forecast **future observations** with uncertainty estimates

---

### **1. Setup and Dependencies**

We install and import the required libraries:
- (**PyMC**)[https://www.pymc.io/] → Bayesian modeling and inference
- **ArviZ** → posterior analysis and visualization
- **Matplotlib** → plotting results


### **2. Defining a Simple Linear State-Space Model**

We start by simulating data from a deterministic linear trend with Gaussian noise:

$$
y_t \sim \mathcal{N}(\alpha + \beta t,\ \sigma^2)
$$

Where:
- $\alpha$ → baseline level at $t=0$
- $\beta$ → slope over time
- $\sigma$ → observation noise
- $y_t$ → observed or simulated data

We define priors for $\alpha$, $\beta$, and $\sigma$, and then specify the model using **PyMC**.

