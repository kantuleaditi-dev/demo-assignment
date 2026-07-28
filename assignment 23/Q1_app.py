# Q1 - Linear Regression on Insurance Dataset
# Step 1: Import Libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np
# Step 2: Load Dataset
df = pd.read_csv("insurance.csv")
# Step 3: Display Dataset
print("First 5 Rows:")
print(df.head())
print("\nDataset Info:")
print(df.info())
print("\nMissing Values:")
print(df.isnull().sum())

# Step 4: Encode Categorical Columns
le = LabelEncoder()
df["sex"] = le.fit_transform(df["sex"])
df["smoker"] = le.fit_transform(df["smoker"])
df["region"] = le.fit_transform(df["region"])

# Step 5: Separate Features and Target
X = df.drop("expenses", axis=1)
y = df["expenses"]

# Step 6: Split Dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Step 7: Create and Train Model
model = LinearRegression()
model.fit(X_train, y_train)

# Step 8: Prediction
y_pred = model.predict(X_test)

# Step 9: Evaluation
print("\nModel Evaluation")
print("MAE :", mean_absolute_error(y_test, y_pred))
print("MSE :", mean_squared_error(y_test, y_pred))
print("RMSE:", np.sqrt(mean_squared_error(y_test, y_pred)))
print("R2 Score:", r2_score(y_test, y_pred))
# Step 10: Compare Actual vs Predicted
result = pd.DataFrame({
    "Actual Charges": y_test.values,
    "Predicted Charges": y_pred
})
print("\nActual vs Predicted:")
print(result.head(10))