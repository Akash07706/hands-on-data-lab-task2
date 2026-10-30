import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load the sample dataset
df = pd.read_csv("student_performance_lab.csv")

print("First five rows:")
print(df.head())

print("\nDataset information:")
df.info()

print("\nMissing values:")
print(df.isnull().sum())

print("\nDescriptive statistics:")
print(df[["study_hours", "attendance_pct", "assignments_completed", "exam_score"]].describe())

print("\nCorrelation matrix:")
print(df[["study_hours", "attendance_pct", "assignments_completed", "exam_score"]].corr())

df["performance_band"] = pd.cut(
    df["exam_score"],
    bins=[0, 49, 69, 84, 100],
    labels=["Needs Improvement", "Average", "Good", "Excellent"]
)

print("\nStudents in each performance band:")
print(df["performance_band"].value_counts().sort_index())

df.to_csv("student_performance_with_bands.csv", index=False)

plt.figure(figsize=(7, 4.5))
plt.scatter(df["study_hours"], df["exam_score"])
plt.title("Study Hours vs Exam Score")
plt.xlabel("Study Hours")
plt.ylabel("Exam Score")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("study_hours_vs_exam_score.png", dpi=160)
print("\nSaved processed CSV and scatter plot.")
