"""
Yuva Intern Task 4 - Data Science Tool Mastery Project
Student Performance Analysis and Score Prediction
Dataset: synthetic practice data generated for learning; not real student records.
Run: python main.py
"""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data" / "student_performance.csv"
OUT = ROOT / "outputs"
OUT.mkdir(parents=True, exist_ok=True)
def main():
    df = pd.read_csv(DATA)
    print("\\n--- DATA PREVIEW ---")
    print(df.head())
    print("\\n--- DATASET INFO ---")
    print(f"Rows: {len(df)}, Columns: {len(df.columns)}")
    print("\\nMissing values by column:")
    print(df.isnull().sum())
    print("\\nDescriptive statistics:")
    print(df.describe().round(2))

    # Basic data validation and cleaning
    df = df.drop_duplicates()
    df = df.dropna()
    df = df[(df["attendance_percent"].between(0, 100)) &
            (df["previous_exam_score"].between(0, 100)) &
            (df["assignment_average"].between(0, 100)) &
            (df["final_exam_score"].between(0, 100))]

    # Exploratory data analysis
    plt.figure(figsize=(7, 4))
    plt.scatter(df["study_hours_per_day"], df["final_exam_score"], alpha=0.7)
    plt.xlabel("Study hours per day")
    plt.ylabel("Final exam score")
    plt.title("Study Time vs Final Exam Score")
    plt.tight_layout()
    plt.savefig(OUT / "study_vs_score.png", dpi=150)
    plt.close()

    plt.figure(figsize=(7, 4))
    plt.hist(df["final_exam_score"], bins=12, edgecolor="black")
    plt.xlabel("Final exam score")
    plt.ylabel("Number of students")
    plt.title("Distribution of Final Exam Scores")
    plt.tight_layout()
    plt.savefig(OUT / "score_distribution.png", dpi=150)
    plt.close()

    # Machine-learning regression model
    X = df.drop(columns=["final_exam_score"])
    y = df["final_exam_score"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42
    )
    model = RandomForestRegressor(
        n_estimators=250, random_state=42, max_depth=6
    )
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    mae = mean_absolute_error(y_test, predictions)
    rmse = mean_squared_error(y_test, predictions) ** 0.5
    r2 = r2_score(y_test, predictions)
    print("\\n--- MODEL EVALUATION (held-out test set) ---")
    print(f"MAE : {mae:.2f}")
    print(f"RMSE: {rmse:.2f}")
    print(f"R2  : {r2:.3f}")

    importance = pd.Series(
        model.feature_importances_, index=X.columns
    ).sort_values()
    ax = importance.plot(kind="barh", title="Random Forest Feature Importance")
    ax.set_xlabel("Feature importance")
    plt.tight_layout()
    plt.savefig(OUT / "feature_importance.png", dpi=150)
    plt.close()

    print("\\n--- KEY SUMMARY ---")
    print(f"Average final score: {df['final_exam_score'].mean():.2f}")
    print(f"Practice pass rate (score >= 40): "
          f"{(df['final_exam_score'] >= 40).mean() * 100:.1f}%")
    print("\\nCharts saved in the outputs/ folder.")

if __name__ == "__main__":
    main()