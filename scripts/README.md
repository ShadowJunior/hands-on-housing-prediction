# Hands-on Housing Prediction — `housing_prediction.ipynb`

This repository contains a small end-to-end example for getting started with a machine learning workflow using a housing dataset. The main artifact is the Jupyter notebook `housing_prediction.ipynb` which walks through loading data, exploratory data analysis, simple feature engineering, and two types of train/test splits (simple and stratified).

This README explains what the notebook does, how to run it, how to reproduce the work on your own machine, and resources for learning Jupyter, Python, and machine learning.

---

## Project overview

Files in this folder:

- `housing_prediction.ipynb` — The Jupyter notebook with the step-by-step analysis.
- `housing_prediction.py` — Original script that was converted into the notebook.
- `housing.csv` — The dataset used by the notebook (expected to be in the same folder).

Goal: provide a clear, runnable example that shows how to start a small ML project: load data, inspect it, do simple preprocessing, and split it for modeling.

---

## Notebook structure (high-level)

The notebook is organized into logical steps. Run cells top-to-bottom (imports first) to avoid NameErrors.

1. Title / Overview (Markdown)
2. Imports (all required Python packages)
3. Step 1 — Load data (reads `housing.csv` from the notebook folder)
4. Step 2 — Explore dataset (info, describe, simple visualizations)
5. Step 3 — Feature engineering (creates `income_cat` by binning median income)
6. Step 4 — Prepare features and target (separates `X` and `y`)
7. Step 5 — Simple train/test split (random split with `train_test_split`)
8. Step 6 — Stratified train/test split (maintains `income_cat` distribution using `StratifiedShuffleSplit`)

Tips:
- Always run the import cell first so functions like `train_test_split` and `StratifiedShuffleSplit` are defined.
- If a cell fails because a variable is undefined, run the preceding cells (or re-run the imports cell).

---

## How to run (local)

The instructions below give two common options: using conda (recommended) or pip + venv.

### Using conda (recommended)

1. Create and activate a conda env (example name: `housing-prediction`):

```bash
conda create -n housing-prediction python=3.11 -y
conda activate housing-prediction
```

2. Install required packages (basic set):

```bash
pip install numpy pandas matplotlib scikit-learn jupyterlab
```

3. Open the notebook in VS Code or JupyterLab:

- To open in JupyterLab (browser):

```bash
jupyter lab
```

- Or open the file `housing_prediction.ipynb` directly in VS Code (make sure the Python interpreter is the `housing-prediction` environment).

4. Run cells top-to-bottom (Shift+Enter to run a cell in Jupyter/VS Code). Start with the imports cell.

### Using pip + venv

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install numpy pandas matplotlib scikit-learn jupyterlab
jupyter lab
```

---

## Expected dependencies

Minimum packages used in the notebook:

- Python 3.8+ (3.11 recommended)
- numpy
- pandas
- matplotlib
- scikit-learn
- jupyterlab (or Jupyter notebook)

You can pin versions and create a `requirements.txt` with:

```bash
pip freeze > requirements.txt
```

---

## Data: `housing.csv`

The notebook expects a file named `housing.csv` in the same folder as the notebook. If you don't have the file, the notebook's data-loading cell will fail with a FileNotFoundError.

If the dataset is missing, either:
- Download or place the CSV into this folder, or
- Modify the load cell to point to the dataset path you have.

---

## Explanation of key steps (brief)

- Data loading: `pd.read_csv(filepath)` — loads the CSV into a pandas DataFrame.
- Exploration: `DataFrame.info()`, `DataFrame.describe()`, and simple histograms/plots show distributions and missing values.
- Feature engineering: Binning `median_income` into `income_cat` is useful for stratified sampling and gives an example of creating derived features.
- Simple split: `train_test_split(X, y, test_size=0.33)` produces a random training and test split.
- Stratified split: `StratifiedShuffleSplit` keeps the distribution of `income_cat` similar across train and test sets — important when a feature correlates with the target.

Why stratified sampling? Without it, rare categories might be under-represented in train or test sets, affecting model evaluation stability.

---

## How to turn this notebook into a repeatable script or package

1. Refactor code into functions (for loading, preprocessing, splitting). Example contract for a load function:

```python
# load_data(path: str) -> pd.DataFrame
# preprocess(df: pd.DataFrame) -> pd.DataFrame
# split_data(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]
```

2. Add a `if __name__ == "__main__"` runner that calls those functions with CLI args (use `argparse`).
3. Add unit tests for core functions using `pytest`.
4. Create a `requirements.txt` or `environment.yml` for reproducible environments.

---

## Testing suggestions

- Use `pytest` to test small, fast units:
  - Test that `load_data` reads a CSV and returns a DataFrame with expected columns.
  - Test that `preprocess` creates `income_cat` with expected bins.
- Example test runner cell (not recommended for production but handy during development):

```python
import pytest
pytest.main(["-q"])  # runs tests in the repo
```

---

## Common troubleshooting

- NameError (e.g., `train_test_split` not defined): run the imports cell first — the notebook loads imports separately from later cells.
- FileNotFoundError on `housing.csv`: ensure the CSV file is present in this folder, or update the path in the load cell.
- Missing package: activate your environment and `pip install` the missing package.

---

## Learning resources (beginner-friendly)

Jupyter / Notebooks:
- Official Jupyter docs: https://jupyter.org/documentation
- JupyterLab: https://jupyterlab.readthedocs.io/en/stable/
- VS Code Jupyter extension docs: https://code.visualstudio.com/docs/datascience/jupyter-notebooks

Python (beginners -> intermediate):
- Official Python tutorial: https://docs.python.org/3/tutorial/
- Real Python tutorials: https://realpython.com/
- Automate the Boring Stuff with Python (book/online): https://automatetheboringstuff.com/

Pandas / Data wrangling:
- Official pandas docs: https://pandas.pydata.org/docs/
- "Python for Data Analysis" (book by Wes McKinney)

Machine Learning (intro to applied ML):
- scikit-learn user guide: https://scikit-learn.org/stable/user_guide.html
- Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow (Aurélien Géron) — excellent practical book
- Coursera: "Machine Learning" by Andrew Ng (classic conceptual course): https://www.coursera.org/learn/machine-learning
- fast.ai Practical Deep Learning courses: https://www.fast.ai/
- Kaggle Learn micro-courses: https://www.kaggle.com/learn

General tips for learning:
- Do small end-to-end projects (data -> preprocess -> model -> evaluate -> iterate).
- Start with scikit-learn for classic ML pipelines, then explore deep learning when needed.
- Practice reading other's notebooks (Kaggle kernels) to learn patterns.

---

## Next steps / enhancements you might want

- Refactor notebook code into functions and add small unit tests.
- Add a `requirements.txt` or `environment.yml` for reproducibility.
- Add a small `Makefile` or helper shell script to create the environment and run the notebook.
- Expand the notebook with a modeling section: train a simple regressor (e.g., RandomForestRegressor), evaluate on the stratified test set, and persist the trained model with `joblib` or `pickle`.

---

## Contact / Attribution

If you want me to expand this README with specific commands tailored to your environment (e.g., conda channels, specific Python version), tell me your preferred environment and I'll add them.

Happy learning! 👩\u200d💻👨\u200d💻
