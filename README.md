# Penguin Flipper Length Prediction & Simpson's Paradox

Predicting penguin flipper length from bill measurements using linear regression — and uncovering a real-world instance of Simpson's Paradox through one-hot encoding of categorical species data.

## Overview

This project explores linear regression on the Palmer Penguins dataset. Beyond building a predictive model for flipper length, it uses model coefficients and visualizations to demonstrate **Simpson's Paradox**: a case where a statistical trend present in aggregated data reverses when the data is broken into subgroups.

## Dataset

The [Palmer Penguins dataset](https://github.com/allisonhorst/palmerpenguins) (loaded via seaborn) contains morphological measurements for three penguin species: Adelie, Chinstrap, and Gentoo.

**Features used:**
- `bill_length_mm` — Bill length in millimeters
- `bill_depth_mm` — Bill depth in millimeters
- `species` — Categorical: Adelie, Chinstrap, or Gentoo
- `sex` — Categorical: Male or Female

**Target:** `flipper_length_mm` — Flipper length in millimeters

Data points with missing values (11 out of ~330) were dropped prior to modeling.

## Methods

### Train/Test Split
Data split 70/30 with `random_state=2024`.

### Numerical Features Only (OLS)
A linear regression model using only `bill_length_mm` and `bill_depth_mm` as inputs. This serves as the baseline before incorporating categorical information.

### One-Hot Encoding + OLS
`species` and `sex` are encoded using `OneHotEncoder` and concatenated with the numerical features. A new OLS model is trained on this expanded feature set.

### Simpson's Paradox Analysis
Comparing coefficients between the Task 2 and Task 3 models reveals a striking sign flip on the `bill_depth_mm` coefficient:

| Model | `bill_depth_mm` coefficient |
|-------|-----------------------------|
| Task 2 (numerical only) | **Negative** |
| Task 3 (with species/sex) | **Positive** |

This is Simpson's Paradox in action. The aggregate trend — deeper bills correlate with shorter flippers — reverses once we condition on species.

## Results

| Model | Test RMSE (mm) |
|-------|----------------|
| Numerical features only (Task 2) | ~6.8 |
| Numerical + species + sex (Task 3) | ~5.1 |

Incorporating species and sex meaningfully improves predictive accuracy.

## Key Findings

**Why does the `bill_depth_mm` coefficient flip sign?**

In the unadjusted model, the negative correlation between bill depth and flipper length is a **confounding effect of species**. Smaller species (like Adelie) tend to have *deeper* bills and *shorter* flippers than larger species (like Gentoo). When looking across all species combined, this creates a spurious negative association. Once species is controlled for via one-hot encoding, the within-species relationship is recovered: within each species, penguins with deeper bills *do* tend to have longer flippers — a positive relationship that makes biological sense.

This is a textbook example of why controlling for confounding variables matters in statistical modeling.

## Tech Stack

- Python 3
- scikit-learn (`LinearRegression`, `OneHotEncoder`, `train_test_split`, `root_mean_squared_error`)
- seaborn (dataset loading and visualization)
- NumPy, pandas

## How to Run

```bash
pip install scikit-learn seaborn pandas numpy jupyter
jupyter notebook penguin.ipynb
```

No external data download needed — the dataset is loaded via seaborn.
