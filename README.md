This repository contains the code for our suite of experiments for the paper "Preconditioned subgradient method for composite optimization:
overparameterization and fast convergence".

## Description

We evaluate recovery of nonnegative least squares, low-rank matrices and tensor sensing, under ill-conditioning and overparameterization
 using a variety of iterative methods, including:

- **Polyak Subgradient**
- **Gauss–Newton**
- **Levenberg–Marquardt (ours)**

All experiments generate synthetic data, initialize factors, run each method for a fixed number of iterations, and save both raw losses and summary plots.

## Requirements

- **Python:** 3.11.11
- **CUDA:** 12.4 (for GPU support)
- **PyTorch:** 2.5.1 (built with CUDA 12.4)
- **NumPy:** 1.26.4

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


## Experiment Data Structures

### `experiment_setups`

A **list of pairs** specifying the problem's regime:

- `r`: rank of factors. Overparameterization when r > r_true.
- `κ`: condition number of ground truth.

Used by both:

```python
run_matrix_tensor_sensing_experiments(..., experiment_setups, ...)
run_nonegative_least_squares_experiments(..., experiment_setups, ...)
```


## Running the Experiments

Open `main.ipynb` and execute the following sections in order:

### Experiment 0: Introduction Example

- **Model:** Symmetric matrix factorization
- **Settings:** `n1=n2=50`, `r_true=2`, `r=5`, `tensor=False`, `symmetric=True`, `Identity=True`. Identity set to False creates an instance of sensing with IID Gaussians properly normalized.
- **Setups:** `[(r,1)]` with `r_true=2`. This is the overparameterized regime.
- **Runner:** `run_matrix_tensor_sensing_experiments(...)` with Polyak stepsizes until divergence
- **Output:** `$save_dir/experiment_results/polyak/`

### Experiment 1: Exact Observations with Polyak Stepsizes

- **Problem:** Nonnegative least squares, matrix and tensor sensing.
- **Setups:** `[(r_true,1),(r,100),(r,1),(r_true,100)]` with `r_true=10, r=100 (r_true=2, r=5) for matrix/tensor` 
- **Methods:** `['Polyak Subgradient','Levenberg-Marquardt (ours)'`. Add competitors if needed
- **Iterations:** 500 or 1000
- **Runner:** `run_nonegative_least_squares_experiments(...)` or `run_matrix_tensor_experiments(...)`. `Identity=true` is factorization problem. 
`loss_ord=1` for l1-norm, `loss_ord=0.5` for l2-norm, `loss_ord=2` for l2-norm squared.
- **Output:** `$save_dir/experiment_results/polyak/`



### Experiment 2: Hyperparameter Sensitivity

- **Focus:** Median iterations vs. γ for hyperparameters `q=0.95,0.96,0.97`
- **Grid:** `gammas=[10^i,...]`, `lambdas=[1e-5]`, `tests=[(2,1),(2,100),(5,100)]`
- **Runner:** Loop over `(corr_level, r,kappa, q, γ, λ)` and call `run_matrix_tensor_experiments.py`, collect medians into `to_be_plotted`, then:
  ```python
  plot_results_sensitivity(to_be_plotted, ...)
  ```
- **Output:** `$save_dir/experiment_results/hyperparameter_sensitivity/`

### Experiment 3: Sensing with Outliers

- **Goal:** Heatmaps of success ratios over `(m, p_fail)`
- **Runner:** `run_matrix_tensor_experiments.py` + `plot_transition_heatmap`
- **Output:** `save_dir/experiment_results/outliers_vs_measurements/`

## Notes

- Feel free to modify any parameter blocks in `main.ipynb` for custom sweeps.

---

*Happy experimenting!*

