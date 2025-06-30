#Preconditioned subgradient for matrix & tensor sensing experiments

This repository contains the code and notebooks for our suite of experiments for our paper "Preconditioned subgradient method for composite optimization:
overparameterization and fast convergence".

## Description

We evaluate recovery of nonnegative least squares, low-rank matrices and tensor sensing, under ill-conditioning and overparameterization
 using a variety of iterative methods, including:

- **Polyak Subgradient**
- **Gauss–Newton**
- **Levenberg–Marquardt (ours)**

All experiments generate synthetic data, initialize factors, run each method for a fixed number of iterations (or until convergence), and save both raw losses and summary plots.

## Requirements

- **Python:** 3.11.11
- **CUDA:** 12.4 (for GPU support)
- **PyTorch:** 2.5.1 (built with CUDA 12.4)
- **NumPy:** 1.26.4

Linear systems are solved via Conjugate Gradient (max 100 iterations, tolerance = 1e-25) using implicit \(\nabla F^T\nabla F\) actions except in nonnegative least squares (direct closed-form computation available).

**System Packages (for LaTeX support in plots and notebooks):**

| Package                   | Version (Ubuntu 22.04 LTS, June 2024) |
|---------------------------|----------------------------------------|
| cm-super                  | 0.3.4-12                               |
| dvipng                    | 1.17-1                                 |
| texlive-latex-extra       | 2021.20220204-1                        |
| texlive-latex-recommended | 2021.20210418-1                        |

## Installation

1. **Clone the repo**
   ```bash
   git clone https://github.com/aglabassi/preconditioned_composite_opti repo
   cd repo
   ```
2. **Install System Dependencies (for LaTeX support, e.g., math rendering in plotting or Jupyter):**
   Install via:
   ```
   bash
   sudo apt install cm-super dvipng texlive-latex-extra texlive-latex-recommended
   ```
3. **Create** and **activate** the environment (Conda recommended):
   ```bash
   conda env create -f environment.yml -n mts
   conda activate mts
   ```
4. **Install** the package in editable mode:
   ```bash
   pip install -e .

## Configuration

Set your output directory before running any code:

```bash
export SAVE_PATH=/path/to/where/you/want/results
```

All logs and plots will be saved under:

```
$SAVE_PATH/experiment_results/...
```

## Experiment Data Structures

### `experiment_setups`

A **list of ****\`\`**** tuples** specifying problem instances:

- `r`: rank or over‑parameterization level
- `κ`: condition number of the measurement operator

Used by both:

```python
run_matrix_tensor_sensing_experiments(..., experiment_setups, ...)
run_nonegative_least_squares_experiments(..., experiment_setups, ...)
```

Each tuple drives generation of data, initialization, and method runs.

### `to_be_plotted` (hyperparameter sensitivity)

A nested dict describing median iteration counts over methods and hyperparameter grids:

```python
Dict[
  ProblemKey,                    # e.g. corruption level or dummy string
  Dict[
    MethodName,                  # e.g. 'Polyak Subgradient'
    np.ndarray(shape=(n_lambdas,n_gammas), dtype=object)
      # each entry = (median_iters, (lower_5%, upper_95%))
  ]
]
```

Fed to:

```python
plot_results_sensitivity(to_be_plotted, ...)
```

## Running the Experiments

Open `main.ipynb` and execute the following sections in order:

### Experiment 0: Introduction Example

- **Model:** Symmetric matrix factorization
- **Settings:** `n1=n2=50`, `r_true=2`, `tensor=False`, `symmetric=True`
- **Runner:** `run_matrix_tensor_sensing_experiments(...)` with Polyak stepsizes until divergence
- **Output:** `$SAVE_PATH/experiment_results/polyak/matrixsym/`

### Experiment 1: Exact Observations with Polyak Stepsizes

- **Problem:** Nonnegative least squares
- **Setups:** `[(r_true,1),(r,100),(r,1),(r_true,100)]` with `r_true=10, r=100`
- **Methods:** `['Polyak Subgradient','Levenberg-Marquardt (ours)','Gauss-Newton']`
- **Iterations:** 500
- **Runner:** `run_nonegative_least_squares_experiments(...)`
- **Output:** `$SAVE_PATH/experiment_results/polyak/nonneg_ls/`

### Experiment 2: Hyperparameter Sensitivity

- **Focus:** Median iterations vs. γ for quantiles `q=0.95,0.96,0.97`
- **Grid:** `gammas=[10^i,...]`, `lambdas=[1e-5]`, `tests=[(2,1),(2,100),(5,100)]`
- **Runner:** Loop over `(corr_level, r,kappa, q, γ, λ)`, collect medians into `to_be_plotted`, then:
  ```python
  plot_results_sensitivity(to_be_plotted, ...)
  ```
- **Output:** `$SAVE_PATH/experiment_results/hyperparameter_sensitivity/`

### Experiment 3: Sensing with Outliers

- **Goal:** Heatmaps of success ratios over `(m, p_fail)`
- **Runner:** `run_matrix_tensor_experiments.py` + `plot_transition_heatmap`
- **Output:** `$SAVE_PATH/experiment_results/outliers_vs_measurements/`

## Notes

- If you need to re-run an experiment without stale logs, delete the corresponding `.pkl` or log files under `experiment_results/` before executing.
- Feel free to modify any parameter blocks in `main.ipynb` for custom sweeps.

---

*Happy experimenting!*

