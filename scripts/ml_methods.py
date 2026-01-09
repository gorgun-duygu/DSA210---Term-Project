# Machine Learning - Anime Rating Prediction
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor

# Folders to store the plots and the summary
script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(script_dir)

csv_path = os.path.join(parent_dir, "kaggle_anime_database.csv")

outputs_dir = os.path.join(parent_dir, "outputs")
reports_dir = os.path.join(parent_dir, "reports")
os.makedirs(outputs_dir, exist_ok=True)
os.makedirs(reports_dir, exist_ok=True)

# Load and clean the data
df = pd.read_csv(csv_path)
print("Data loaded successfully for ML!")

# Keep rows where rating exists (target)
df = df.dropna(subset=["rating"])

# Convert episodes and members to numeric
df["episodes"] = pd.to_numeric(df["episodes"], errors="coerce")
df["members"] = pd.to_numeric(df["members"], errors="coerce")

# Extract main genre
df["main_genre"] = df["genre"].astype(str).str.split(",").str[0].str.strip()

# Drop rows with missing feature values
df = df.dropna(subset=["episodes", "members", "type", "main_genre"])

# Define Features (x) and Target (y)
X = df[["episodes", "members", "type", "main_genre"]]
y = df["rating"]

categorical_features = ["type", "main_genre"]
numeric_features = ["episodes", "members"]

# One-hot encode categorical features and pass numeric features through
preprocessor = ColumnTransformer(
    transformers=[
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features),
        ("num", "passthrough", numeric_features)
    ]
)

# Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Method 1: Linear Regression
print("Training Linear Regression model...")
lr_model = Pipeline(steps=[
    ("preprocess", preprocessor),
    ("model", LinearRegression())
])

lr_model.fit(X_train, y_train)
y_pred_lr = lr_model.predict(X_test)

rmse_lr = np.sqrt(mean_squared_error(y_test, y_pred_lr))
r2_lr = r2_score(y_test, y_pred_lr)

# Method 2: Random Forest
print("Training Random Forest model...")
rf_model = Pipeline(steps=[
    ("preprocess", preprocessor),
    ("model", RandomForestRegressor(
        n_estimators=200,
        random_state=42,
        n_jobs=-1
    ))
])

rf_model.fit(X_train, y_train)
y_pred_rf = rf_model.predict(X_test)

rmse_rf = np.sqrt(mean_squared_error(y_test, y_pred_rf))
r2_rf = r2_score(y_test, y_pred_rf)

# Save results to a text file
report_path = os.path.join(reports_dir, "ml_results.txt")
with open(report_path, "w", encoding="utf-8") as f:
    f.write("Machine Learning Results: Anime Rating Prediction\n")
    f.write("Target: rating\n")
    f.write("Features: episodes, members, type, main_genre\n")
    f.write(f"Dataset size used for ML methods application: {len(df)} rows\n")

    f.write("Method 1: Linear Regression\n")
    f.write(f"RMSE: {rmse_lr:.4f}\n")
    f.write(f"R^2 : {r2_lr:.4f}\n\n")

    f.write("Method 2: Random Forest Regressor\n")
    f.write(f"RMSE: {rmse_rf:.4f}\n")
    f.write(f"R^2 : {r2_rf:.4f}\n\n")

# Save a plot of Actual vs Predicted for Random Forest
plt.figure(figsize=(7,5))
plt.scatter(y_test, y_pred_rf, alpha=0.35, s=12)
plt.title("Actual vs Predicted Ratings (Random Forest)")
plt.xlabel("Actual Rating")
plt.ylabel("Predicted Rating")

# line y=x to show the perfect prediction
min_val = min(y_test.min(), y_pred_rf.min())
max_val = max(y_test.max(), y_pred_rf.max())
plt.plot([min_val, max_val], [min_val, max_val], color='red', linestyle='--')

plot_path = os.path.join(outputs_dir, "ml_actual_vs_predicted.png")
plt.tight_layout()
plt.savefig(plot_path, dpi=150)
plt.close()

print("ML methods application completed!")
print("ML methods application results are saved in the reports folder and the created plots are saved in the outputs folder.")