import pandas as pd
import numpy as np
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.model_selection import train_test_split
import shap
import joblib
import pickle
import matplotlib.pyplot as plt

# Load data
df = pd.read_csv("./data/cleaned.csv")

# Split data
X, y = df.iloc[:, :-1], df.iloc[:, -1]  # Corrected column indexing
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train model
model = ExtraTreesRegressor(random_state=42)
model.fit(X_train, y_train)

# Save the model and test data for later use
joblib.dump(model, "et_regressor_model.joblib")
with open("X_test.pkl", "wb") as f:
    pickle.dump(X_test, f)

# SHAP analysis
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_test)

# Create and save SHAP summary plot
shap.summary_plot(shap_values, X_test, plot_type="bar", show=False)
plt.savefig("shap_summary_plot.png")

print("02_ml_runner.py executed successfully. Model and data saved.")
