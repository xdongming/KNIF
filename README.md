# KNIF: Koopman Normalizing Invertible Flow

> A unified Koopman modeling framework that couples **conditional normalizing flows** with **global linear evolution** for long‑horizon prediction, reconstruction, and interpretability on nonlinear dynamical systems.

<p align="center">
  <b>Non‑invertibility · Input non‑closure · Distributional non‑identifiability → One unified solution</b>
</p>

---

## Overview

**KNIF (Koopman Normalizing Invertible Flow)** combines a two‑stage observable mapping with Koopman‑style linear evolution:

1. An **invertible, reconstructible flow** maps \((x, u)\) to a high‑capacity latent observable space (optionally preceded by Chebyshev/Conv1D expansions for delays and local features).  
2. A **global linear operator** \(K\) advances the latent state, \(z_{t+1} pprox K z_t\).  
3. The **inverse flow** decodes back to the original space, closing the loop for reconstruction and multi‑step forecasting.

This repository includes KNIF’s reference implementation, notebooks, and baseline methods (DeepKoopman, EDMD‑DL, FlowDMD) together with ablation and robustness studies.

---

## Key Features

- **Invertible & reconstructible observables** via normalizing flows  
- **Conditional input handling** for input‑driven systems  
- **Distribution modeling** (non‑Gaussian noise / shifts) built into the flow  
- **Koopman linear evolution** in the latent space for interpretability  
- **Ablations and baselines** for fair comparison and analysis

---

## Repository Structure

> The exact layout may evolve; use the folders relevant to your experiments.

```
KNIF/
├─ DeepKoopman/          # DeepKoopman baseline
├─ EDMD-DL/              # EDMD-DL (dictionary learning) baseline
├─ Flowdmd/              # FlowDMD baseline
├─ Real Robustness/      # Experiments under real-world disturbances
├─ Robot/ Wave/ Wind/    # Task- or data-specific examples
├─ ablation/             # Ablations (replace/remove modules)
├─ challenge/            # Extra challenge tasks
├─ extension/            # Extensions and exploratory work
├─ hidden/               # Shared modules (e.g., observable dictionaries/solvers)
├─ input/                # Input-modeling examples
├─ lightning_logs/       # Training logs (if using Lightning)
├─ ode/ pde/ sde/        # ODE / PDE / SDE notebooks and demos
├─ plot/ sample/ noise/  # Plotting, samples, noise utilities
└─ README.md
```

---

## Installation

We recommend **Python 3.9+** and a clean virtual/conda environment.

```bash
# 1) Clone
git clone https://github.com/xdongming/KNIF.git
cd KNIF

# 2) Create environment (example)
conda create -n knif python=3.10 -y
conda activate knif

# 3) Core utilities
pip install numpy scipy matplotlib jupyterlab tqdm scikit-learn

# 4a) If you run TF/Keras-based components (some shared modules)
pip install "tensorflow==2.*" keras

# 4b) If you run PyTorch/Lightning baselines or reproduce logs
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121   # adjust to your CUDA
pip install lightning optuna
```

> Note: Some subfolders rely on TensorFlow/Keras (e.g., certain dictionary/solver utilities under `hidden/`), while others use PyTorch/Lightning. Install only what you need.

---

## Quick Start

### A) Run notebooks
1. Open a notebook in `ode/`, `sde/`, `pde/`, or `ablation/`.  
2. Execute cells top‑down: data → train (estimate **K**) → forecast/reconstruct → plot.  
3. For baselines, use the notebooks/scripts under `DeepKoopman/`, `EDMD-DL/`, and `Flowdmd/`.

### B) Minimal pseudo‑code
```python
# Data: (x_t, u_t) → (x_{t+1}, u_{t+1})
x, u = load_sequences(...)

# 1) Conditional invertible flow: (x, u) -> z
z = g_theta.encode(x, u)  # optional: Chebyshev/Conv1D for delays/features

# 2) Estimate linear operator K:  z_{t+1} ≈ K z_t  (DMD / regularized OLS)
K = estimate_linear_operator(z[:, :-1, :], z[:, 1:, :], reg=1e-6)

# 3) Closed-loop prediction & reconstruction
z_pred = rollout_linear(K, z0=z[:, 0, :], steps=T)
x_pred = g_theta.decode(z_pred, cond=u)

# 4) Loss
loss = mse(x_pred, x) + lambda_reg * regularizers(...)
optimize(loss)
```

**Tips**
- Treat inputs as **conditions** in the flow or concatenate them after feature lifting.  
- Ensure **consistency** by supervising reconstruction in the original space via the inverse map.  
- Evaluate **long‑horizon rollouts** for stability and error propagation.

---

## Experiments

- **Ablations** (`ablation/`): remove invertible expansion, remove conditioning, replace flow with AE, etc.  
- **Robustness** (`Real Robustness/`): noise, distribution shift, real‑world disturbances.  
- **SDE/ODE/PDE** (`sde/`, `ode/`, `pde/`): stochastic/ordinary/partial differential system demos.  
- **Baselines** (`DeepKoopman/`, `EDMD-DL/`, `Flowdmd/`): for side‑by‑side comparisons.

To batch experiments, you can extract training cells from notebooks into scripts or use your Lightning/PyTorch trainer.

---

## Design Rationale (Why KNIF)

- **Invertibility & reconstruction**: flow architectures guarantee bijective mapping and faithful decoding.  
- **Input closure**: incorporate inputs \(u\) as conditions/features to close the evolution loop.  
- **Distribution modeling**: flows naturally capture non‑Gaussian stochasticity and shifts.  
- **Interpretability**: linear latent evolution offers Koopman‑style modal insights.

---

## FAQ

- **Do I need a GPU?** Not strictly; most demos run on CPU. Large tasks benefit from a GPU.  
- **Do I need both TF and Torch?** No. Install only what your chosen subfolders require.  
- **Where is the data?** Example data are usually included/generated in each folder; any extras will be documented locally.

---

## Citation

If you use KNIF in your research or products, please cite:

```bibtex
@misc{KNIF,
  title        = {Koopman Normalizing Invertible Flow: A Unified Framework for Nonlinear Modeling with Structural Consistency},
  author       = {X. Dongming and collaborators},
  howpublished = {GitHub Repository},
  year         = {2025},
  url          = {https://github.com/xdongming/KNIF}
}
```

---

## License

This project is released under the **MIT License**. See the repository license file for details.
