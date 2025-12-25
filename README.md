# Loan Approval Bias Detection

An end-to-end machine learning project focused on detecting demographic bias in loan approval outcomes using Python.

---

## 📌 Project Overview

This project analyzes a loan approval dataset to:
- Clean and preprocess real-world data
- Encode categorical variables
- Train a machine learning model to predict loan approval
- Analyze potential bias in predictions across demographic groups such as **Gender** and **Marital Status**

The goal is not only to build an accurate model, but also to **evaluate fairness** in its predictions.

---

## 🛠️ Tech Stack

- Python
- Pandas
- NumPy
- Matplotlib
- Scikit-learn

---

## 📂 Dataset

- Source: Kaggle (Loan Approval Dataset)
- Target Variable: `Loan_Status`
- Sensitive Attributes:
  - Gender
  - Marital Status

---

## 🔄 Project Workflow

1. **Data Cleaning**
   - Removed missing values
   - Prepared clean dataset (`loan_cleaned.csv`)

2. **Encoding & Bias Visualization**
   - Converted categorical variables to numerical form
   - Visualized approval rates across demographic groups

3. **Machine Learning Model**
   - Logistic Regression model trained on encoded data
   - Model accuracy evaluated

4. **Bias Analysis**
   - Compared predicted approval rates across:
     - Gender
     - Marital Status

---

## 📊 Results

- Model Accuracy: ~78%
- Approval rates were analyzed to identify potential demographic bias
- Results show small variations across groups, highlighting the importance of fairness evaluation

---

## 📁 Files in Repository

- `data_cleaning.py` – Data preprocessing
- `encoding_bias_analysis.py` – Encoding & bias visualization
- `ml_model_bias_check.py` – ML model training & bias evaluation
- `loan.csv` – Original dataset
- `loan_cleaned.csv` – Cleaned dataset
- `loan_encoded.csv` – Encoded dataset

---

## 🎯 Key Learning Outcomes

- Practical machine learning pipeline
- Importance of fairness and bias detection in ML systems
- Hands-on experience with real-world data

---

## 🚀 Future Improvements

- Apply fairness-aware ML techniques
- Use additional sensitive attributes
- Try advanced models for comparison

---

## 👤 Author

Sneha 
