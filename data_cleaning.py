import pandas as pd

# Load dataset
df = pd.read_csv(r"C:\Users\hp\Desktop\loan\loan.csv.csv")

# Drop Loan_ID (not useful)
df.drop("Loan_ID", axis=1, inplace=True)

# Fill missing categorical values with mode
categorical_cols = [
    "Gender", "Married", "Dependents",
    "Self_Employed", "Credit_History"
]

for col in categorical_cols:
    df[col].fillna(df[col].mode()[0], inplace=True)

# Fill numerical missing values with median
df["LoanAmount"].fillna(df["LoanAmount"].median(), inplace=True)
df["Loan_Amount_Term"].fillna(df["Loan_Amount_Term"].median(), inplace=True)

# Final check
print(df.isnull().sum())
print(df.head())

# Save cleaned data
df.to_csv("loan_cleaned.csv", index=False)
