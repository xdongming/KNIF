# Repository Overview and Usage

This document summarizes the repository layout and provides simple, generic usage examples (no implementation details).

---

## Directory Structure (top‑level)

```
<root>/
├─ ablation/             # notebooks and examples for ablation studies
├─ challenge/            # challenges description
├─ DeepKoopman/          # baseline (kept as existing folder name)
├─ EDMD-DL/              # baseline (kept as existing folder name)
├─ extension/            # extensions and exploratory work
├─ Flowdmd/              # baseline (kept as existing folder name)
├─ hidden/               # hidden robustness studies
├─ input/                # input systems examples
├─ lightning_logs/       # logs (if used)
├─ ode/                  # ordinary differential system notebooks/examples
├─ pde/                  # partial differential system notebooks/examples
├─ plot/                 # plotting utilities and figures
├─ Real Robustness/      # robustness experiments
├─ Robot/ Wave/ Wind/    # real-world examples
├─ sample/               # sample robustness studies
├─ sde/                  # stochastic differential system notebooks/examples
├─ noise/                # noise robustness studies
├─ README.md
└─ requirements.txt
```

> Folder names are shown as they exist in the repository; use only the parts relevant to your work.

---

## Setup (generic)

We recommend Python 3.11 and an isolated virtual/conda environment.

```bash
# Clone (replace with the appropriate repository URL)
git clone <REPO_URL>
cd <repo-folder>

# Create environment (example)
conda create -n proj python=3.11 -y
conda activate proj

# Install minimal tools to open and run notebooks
pip install -r requirements.txt
```

---

## Running Notebooks (generic)

1. Launch Jupyter:
   ```bash
   jupyter lab
   ```
2. Open a notebook under folders such as `ode/`, `sde/`, `pde/`, `ablation/`, or `Real Robustness/`.
3. Execute cells from top to bottom.

> If a notebook expects data, place files under a convenient path (e.g., `data/`) and adjust paths in the first cell.

---

## License

See the repository license file if present.
