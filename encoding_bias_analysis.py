import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder

# Load cleaned data
df = pd.read_csv("loan_cleaned.csv")

# Encode categorical columns
le = LabelEncoder()
categorical_cols = df.select_dtypes(include='object').columns

for col in categorical_cols:
    df[col] = le.fit_transform(df[col])

# Save encoded data
df.to_csv("loan_encoded.csv", index=False)

# ---- BIAS ANALYSIS ----

# Gender vs Loan Approval
gender_bias = df.groupby("Gender")["Loan_Status"].mean()
print("\nLoan Approval Rate by Gender:")
print(gender_bias)

# Marital Status vs Loan Approval
married_bias = df.groupby("Married")["Loan_Status"].mean()
print("\nLoan Approval Rate by Marital Status:")
print(married_bias)

# Plot Gender Bias
gender_bias.plot(kind="bar", title="Loan Approval Rate by Gender")
plt.ylabel("Approval Rate")
plt.tight_layout()
plt.show()

# Plot Marital Bias
married_bias.plot(kind="bar", title="Loan Approval Rate by Marital Status")
plt.ylabel("Approval Rate")
plt.tight_layout()
plt.show()
