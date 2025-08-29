# Latent Linear Dynamics with Invertible Flows

> A compact framework that combines **conditional invertible flows** with **global linear evolution** in a latent space for long‑horizon forecasting, reconstruction, and analysis of dynamical systems.

This README is intentionally **anonymized** for double‑blind review: it does not include author names, affiliations, lab names, paper titles, or repository owner information.

---

## Overview

The framework maps observed states (and optional inputs) into an **invertible latent space**, evolves them with a **single linear operator**, and decodes back to the original space. This closes the loop for reconstruction while enabling **interpretable linear dynamics** in the latent space. The repository includes reference notebooks, utilities, and several baselines for comparison.

Core ideas (high level):
- **Conditional invertible mapping** from \((x, u)\) to latent observables \(z\).
- **Linear latent evolution** \(z_{t+1} \approx K z_t\) estimated by standard numerical methods (e.g., DMD/regularized OLS).
- **Inverse mapping** to reconstruct trajectories in the original space and support long‑horizon rollouts.
- Optional feature lifting (e.g., Chebyshev/Conv1D with delays) for partially observed settings.

---

## Repository Structure

> The exact layout may evolve. Use the folders relevant to your experiments.

```
<root>/
├─ DeepKoopman/          # baseline (as-is folder name)
├─ EDMD-DL/              # baseline (dictionary learning)
├─ Flowdmd/              # baseline (flow-based DMD)
├─ Real Robustness/      # experiments under real-world disturbances
├─ Robot/ Wave/ Wind/    # task- or data-specific examples
├─ ablation/             # ablations (replace/remove modules)
├─ challenge/            # additional challenge tasks
├─ extension/            # extensions and exploratory work
├─ hidden/               # shared modules (e.g., observable dictionaries/solvers)
├─ input/                # input-modeling examples
├─ lightning_logs/       # training logs (if using Lightning)
├─ ode/ pde/ sde/        # ODE / PDE / SDE notebooks and demos
├─ plot/ sample/ noise/  # plotting, samples, noise utilities
└─ README.md
```

---

## Installation

We recommend **Python 3.9+** and a clean virtual/conda environment.

```bash
# 1) Clone (use your anonymous repository URL)
git clone <ANON_REPO_URL>
cd <repo-folder>

# 2) Create environment (example)
conda create -n anon_dyn python=3.10 -y
conda activate anon_dyn

# 3) Core utilities
pip install numpy scipy matplotlib jupyterlab tqdm scikit-learn

# 4a) If you run TF/Keras-based components (some shared modules)
pip install "tensorflow==2.*" keras

# 4b) If you run PyTorch/Lightning baselines or reproduce logs
pip install torch torchvision torchaudio            # choose wheel per your CUDA
pip install lightning optuna
```

> Note: Some subfolders rely on TensorFlow/Keras (e.g., certain dictionary/solver utilities under `hidden/`), while others use PyTorch/Lightning. Install only what you need.

---

## Quick Start

### A) Run notebooks
1. Open a notebook in `ode/`, `sde/`, `pde/`, or `ablation/`.  
2. Execute cells top‑down: data → train (estimate **K**) → forecast/reconstruct → plot.  
3. For baselines, use the notebooks/scripts under `DeepKoopman/`, `EDMD-DL/`, and `Flowdmd/` (as named in the repository).

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
- Treat inputs as **conditions** in the flow or concatenate after feature lifting.  
- Ensure **consistency** by supervising reconstruction in the original space via the inverse map.  
- Evaluate **long‑horizon rollouts** for stability and error propagation.

---

## Experiments

- **Ablations** (`ablation/`): remove invertible expansion, remove conditioning, replace flow with autoencoder, etc.  
- **Robustness** (`Real Robustness/`): noise, distribution shift, and real‑world disturbances.  
- **SDE/ODE/PDE** (`sde/`, `ode/`, `pde/`): stochastic/ordinary/partial differential system demos.  
- **Baselines** (`DeepKoopman/`, `EDMD-DL/`, `Flowdmd/`): side‑by‑side comparisons.

To batch experiments, you can extract training cells from notebooks into scripts or use your Lightning/PyTorch trainer.

---

## Reproducibility

- Fix random seeds where appropriate.  
- Save model checkpoints and training configurations.  
- Prefer fixed data splits for fair baseline comparisons.

---

## License

Released under the **MIT License**. See the repository license file for details.

---

## Anonymity Note

This README intentionally avoids personal names, affiliations, lab names, specific paper titles (including any long‑form expansions), and direct links tied to identities. Keep it anonymized during review.
