"""
Generates a synthetic loan approval dataset with intentional bias.
Gender bias: females are approved at a lower rate even with similar qualifications.
"""

import numpy as np
import pandas as pd

np.random.seed(42)

N = 1000

age = np.random.randint(22, 60, N)
income = np.random.randint(20000, 120000, N)
credit_score = np.random.randint(300, 850, N)
employment_years = np.random.randint(0, 30, N)
gender = np.random.choice([0, 1], N)  # 0 = Female, 1 = Male

# Base approval probability from financial features
base_prob = (
    0.3 * (credit_score - 300) / 550 +
    0.3 * (income - 20000) / 100000 +
    0.2 * (employment_years / 30) +
    0.1 * (age - 22) / 38
)
base_prob = np.clip(base_prob, 0, 1)

# Inject gender bias: females get a 20% penalty
bias_penalty = np.where(gender == 0, 0.20, 0.0)
biased_prob = np.clip(base_prob - bias_penalty, 0, 1)

approved = (np.random.rand(N) < biased_prob).astype(int)

df = pd.DataFrame({
    "age": age,
    "income": income,
    "credit_score": credit_score,
    "employment_years": employment_years,
    "gender": gender,  # 0=Female, 1=Male
    "approved": approved
})

df.to_csv("data/loan_data.csv", index=False)
print(f"Dataset saved: {len(df)} rows")
print(df["approved"].value_counts())
print("\nApproval rate by gender:")
print(df.groupby("gender")["approved"].mean().rename({0: "Female", 1: "Male"}))
