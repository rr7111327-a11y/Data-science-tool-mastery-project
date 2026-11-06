# Student Performance Analysis and Score Prediction

## Yuva Intern - Task 4: Data Science Tool Mastery Project

This project demonstrates practical use of Python, NumPy, Pandas, Matplotlib and
Scikit-learn through data inspection, cleaning, exploratory analysis,
visualization and a regression model.

**Important:** `data/student_performance.csv` is a synthetic learning dataset
generated with a fixed random seed. It is not real student data and should not
be used for decisions about actual students.

## Run in VS Code

1. Install Python 3.10 or later.
2. Open this project folder in VS Code.
3. Open Terminal -> New Terminal.
4. (Recommended) Create a virtual environment:
   - Windows: `py -m venv .venv` then `.venv\\Scripts\\activate`
   - macOS/Linux: `python3 -m venv .venv` then `source .venv/bin/activate`
5. Install libraries: `python -m pip install -r requirements.txt`
6. Run: `python main.py`

If `python` does not work on Windows, try `py main.py`.

## Project files

- `main.py` - complete analysis and model code
- `data/student_performance.csv` - practice dataset
- `outputs/` - generated charts
- `requirements.txt` - required Python packages

## Model evaluation (reproducible run)

- MAE: 6.25
- RMSE: 7.59
- R-squared: 0.837

Metrics are for this synthetic dataset and this fixed train/test split.
