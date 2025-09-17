# Logistic Regression on California Housing Dataset
# We'll classify houses as "Expensive" (1) if price > median value else "Cheap" (0)

import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Load Dataset
housing = fetch_california_housing(as_frame=True)
df = housing.frame

# Create binary target variable
median_price = df['MedHouseVal'].median()
df['PRICE_CATEGORY'] = np.where(df['MedHouseVal'] > median_price, 1, 0)  # 1 = Expensive, 0 = Cheap

print(df[['MedHouseVal', 'PRICE_CATEGORY']].head())

# Features & Target
X = df.drop(['MedHouseVal', 'PRICE_CATEGORY'], axis=1)
y = df['PRICE_CATEGORY']

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model
log_reg = LogisticRegression(max_iter=2000)  # increase iterations for convergence
log_reg.fit(X_train, y_train)

# Prediction
y_pred = log_reg.predict(X_test)

# Evaluation
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Cheap','Expensive'], yticklabels=['Cheap','Expensive'])
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Logistic Regression Confusion Matrix")
os.makedirs("../outputs", exist_ok=True)
plt.savefig("../outputs/logistic_regression_cm.png")
plt.show()
