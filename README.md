# DSA5205 Project 1

This repository contains the code for Tasks 1--4 of DSA5205 Project 1.
All commands below should be run from the project root. The scripts use fixed
random seeds, relative paths, YAML configuration files, and timestamped output
directories.

## Environment

The experiments were run with Python 3.10.22. Create and activate an
environment, then install the required packages:

```powershell
conda create -n dsa5205-project1 python=3.10
conda activate dsa5205-project1
python -m pip install -r requirements.txt
```

The Python dependencies and tested versions are recorded in
`requirements.txt`. 

## Task 1: Misspecified Ridge Simulation

Parameters are stored in `task1_2/task1.yaml`. Run:

```powershell
python task1_2/task1.py
```

The script creates `task1_2/task1_result_<timestamp>/` containing:

- `results.csv`;
- plots of test R-squared, expected timing return, Sharpe ratio, and
  coefficient norm against observed complexity.

The reported experiment uses 50 replications, so this run can take several
minutes.

## Task 2: Lasso on the Task 1 Data

Parameters are stored in `task1_2/task2.yaml`. The data-generating process,
seeds, splits, feature permutation, and observed-complexity grid match Task 1.
Run:

```powershell
python task1_2/task2.py
```

The script creates `task1_2/task2_result_<timestamp>/` containing
`results.csv` and the corresponding Lasso plots.

## Task 3: Predictions for A, B, and C

Place the six files supplied for Task 3 in `task3/data/` with the following
names:

```text
pairA_train.csv
pairA_test_features.csv
pairB_train.csv
pairB_test_features.csv
pairC_train.csv
pairC_test_features.csv
```

Each script uses the final 20% of the labeled observations as a chronological
validation block, selects the parameter that maximizes validation
`r2_paper`, refits on all labeled observations, and predicts the public test
features. The parameter search rounds and model comparison are documented in
the report. The exact final settings are given below.

### Data set A: Ridge

Set `task3/ridge.yaml` to:

```yaml
dataset: A
validation_fraction: 0.20
selection_metric: r2_paper

ridge_z_grid:
  - 3.65
  - 3.70
  - 3.75
  - 3.80
  - 3.85
  - 3.90
  - 3.95
```

Run:

```powershell
python task3/ridge.py
```

The selected value should be `z = 3.75`.

### Data set B: Lasso

Set the data set and grid in `task3/lasso.yaml` to:

```yaml
dataset: B
validation_fraction: 0.20
selection_metric: r2_paper
seed: 5205

lasso_alpha_grid:
  - 0.1200
  - 0.1205
  - 0.1210
  - 0.1215
  - 0.1220
  - 0.1225
  - 0.1230
  - 0.1235
  - 0.1240

lasso_max_iter: 5000
lasso_tolerance: 0.0001
```

Run:

```powershell
python task3/lasso.py
```

The selected value should be `alpha = 0.1215`.

### Data set C: Lasso

Set the data set and grid in `task3/lasso.yaml` to:

```yaml
dataset: C
validation_fraction: 0.20
selection_metric: r2_paper
seed: 5205

lasso_alpha_grid:
  - 0.1850
  - 0.1855
  - 0.1860
  - 0.1865
  - 0.1870
  - 0.1875
  - 0.1880
  - 0.1885
  - 0.1890
  - 0.1895
  - 0.1900

lasso_max_iter: 5000
lasso_tolerance: 0.0001
```

Run:

```powershell
python task3/lasso.py
```

The selected value should be `alpha = 0.1865`.

Each Task 3 run creates a timestamped result directory containing:

- `validation_results.csv`;
- one validation plot for each metric;
- `predictions_A.csv`, `predictions_B.csv`, or `predictions_C.csv`.

The prediction CSV contains exactly the columns `t,yhat`; its row order is
copied from the corresponding public test file. 

`task3/nn.py` and `task3/nn.yaml` contain the neural-network candidate used in
the model comparison. They are not used for the final A/B/C predictions.

## Task 4: Neural-Network Training Dynamics

Parameters are stored in `task4/task4.yaml`. Run:

```powershell
python task4/task4.py
```

The script implements the two-layer network and SGD updates directly with
NumPy. It runs five replications for noise levels from 0.0 to 0.5 and creates
`task4/task4_result_<timestamp>/` containing:

- `results.csv` with every recorded checkpoint;
- `best_checkpoints.csv` selected by validation half-MSE;
- `training_dynamics.png`;
- `financial_metrics.png`.

With the reported configuration, the run takes about one minute on the test
machine.
