
# Overview

In recent years, there has been growing concern about the risks associated with data usage, particularly when it comes to **data sharing**. This has led to the introduction of stricter **data management regulations**, which have had a significant impact on scientific research. The effect is especially pronounced in the **biomedical field**, where the sensitivity of the data makes conducting **multi-institutions studies** more challenging.

To address these challenges, [Federated Learning (FL)](https://en.wikipedia.org/wiki/Federated_learning) has gained popularity. FL enables multiple parties to collaboratively train a shared **machine learning model** using their local data **without directly sharing the data itself**. This enhances **privacy** and **security**, as highlighted in the [original FL paper](https://arxiv.org/pdf/1602.05629). The typical setup involves a central server that collects **non-sensitive information** from each data-holding site (e.g., parameters from a locally trained model) and aggregates them into a **global model**.

In this project, I demonstrate the mechanics of a **federated workflow** using [Flower framework](https://flower.ai/), an easy-to-use Python-based library for federated learning. As a simple, didactic example, I compute a weighted mean of per-site gene variances across institutions (clients). This serves as a minimal, illustrative implementation of highly variable gene selection (selecting the top 2000 genes).

# Methods


## Set Up the Environment

The `fl-course-env.yaml` file defines a Conda environment with all the required dependencies for this project.

### Create the environment:
```bash
conda env create -f fl-course-env.yaml
```

### Activate the environment:
```bash
conda activate fl-course-env
```

### Run the FL project using Flower:
```bash
flwr run .
```

# Results

Both the traditional/centralized and federated learning approach provide the same results (results_comparison.ipynb).