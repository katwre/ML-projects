# Introduction to State-Space Models (SSMs) with PyMC

This notebook is a hands-on introduction to **State-Space Models (SSMs)** using [PyMC](https://www.pymc.io/), [ArviZ](https://python.arviz.org/), and [Matplotlib](https://matplotlib.org/).  
It walks through the Bayesian workflow step by step - from simple models to more flexible, realistic ones.

---

## **What's the content:**

- Understand the difference between **deterministic** and **stochastic** trends
- Define **linear Gaussian state-space models**
- Simulate data and perform **prior predictive checks**
- Infer **latent states** and **model parameters** using Bayesian inference
- Evaluate models via **posterior predictive checks**
- Forecast **future observations** with quantified uncertainty
- Extend SSMs by **stacking multiple components** (trend + seasonality + noise)

---

## **Notebook Overview**

### **1. Simple Linear Model**
We start with a **deterministic trend model**:

$$
y_t \sim \mathcal{N}(\alpha + \beta t,\ \sigma^2)
$$

- Simulate synthetic data.
- Sample from the **prior predictive distribution**.
- Fit the model and visualize the inferred line.

---

### **2. Thinking in Systems**
We move from fixed regression to **state-space thinking**:
- Introduce **latent states** $\mu_t$ that evolve over time.
- Define a **Gaussian Random Walk** for $\mu_t$:

$$
\mu_t = \mu_{t-1} + \eta_t, \qquad \eta_t \sim \mathcal{N}(0, \tau^2)
$$

- Separate:
    - **State equation** → hidden dynamics  
    - **Observation equation** → noisy measurements

This is our first **true state-space model**.

---

### **3. Posterior Inference & Predictive Checks**
- Use **MCMC sampling** to infer $\mu_t$ and noise levels.
- Perform **posterior predictive checks** to validate how well the model explains the data.
- Visualize credible intervals and uncertainty bands.

---

### **4. Forecasting Future Observations**
- Extend the time horizon beyond the observed data.
- Predict **future values** using the posterior distribution.
- Quantify **forecast uncertainty** with HDIs.

---

### **5. Mamma Mia - Stacking Components**
In the final section, we build a **richer SSM** by **combining multiple components**:

$$
y_t = \mu_t + s_t + \varepsilon_t
$$

- $\mu_t$ → latent trend (random walk)
- $s_t$ → seasonal or smooth variation
- $\varepsilon_t$ → measurement noise

This model:
- Adapts more dynamically to changing patterns.
- Produces **smarter forecasts**.
- Provides a **full posterior distribution** for each component.

---

## **Summary**

- SSMs separate **hidden dynamics** from **observations**.
- We move from **deterministic regression** → **probabilistic latent models**.
- Bayesian inference lets us:
    - Estimate hidden states $\mu_t$.
    - Quantify uncertainty.
    - Generate forecasts with credible intervals.

---

This notebook provides an intuitive foundation for working with **Bayesian state-space models** and sets the stage for more advanced topics like Kalman filters, hierarchical SSMs, and deep probabilistic forecasting.
