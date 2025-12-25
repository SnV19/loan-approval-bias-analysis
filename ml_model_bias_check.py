import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Load encoded data
df = pd.read_csv("loan_encoded.csv")

# Features & target
X = df.drop("Loan_Status", axis=1)
y = df["Loan_Status"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train model
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# Accuracy
y_pred = model.predict(X_test)
print("Model Accuracy:", accuracy_score(y_test, y_pred))

# Add predictions to full dataset
df["Predicted_Status"] = model.predict(X)

# ---- BIAS CHECK ON MODEL OUTPUT ----

# Gender bias in predictions
gender_bias_model = df.groupby("Gender")["Predicted_Status"].mean()
print("\nPredicted Approval Rate by Gender:")
print(gender_bias_model)

# Marital status bias in predictions
married_bias_model = df.groupby("Married")["Predicted_Status"].mean()
print("\nPredicted Approval Rate by Marital Status:")
print(married_bias_model)
