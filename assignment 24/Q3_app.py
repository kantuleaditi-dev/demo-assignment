import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor

from sklearn.metrics import r2_score

# ===========================
# Load Dataset
# ===========================
df = pd.read_csv("Housing.csv")

# Fill Missing Values
df["total_bedrooms"] = df["total_bedrooms"].fillna(df["total_bedrooms"].median())

# Remove Duplicates
df = df.drop_duplicates()

# Convert Categorical Column
df = pd.get_dummies(df, drop_first=True)

# Features and Target
X = df.drop("median_house_value", axis=1)
y = df["median_house_value"]

# Train Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.20,
    random_state=42
)

# Feature Scaling
scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# ===========================
# Models
# ===========================
models = {
    "Linear Regression": LinearRegression(),
    "Decision Tree Regressor": DecisionTreeRegressor(random_state=42),
    "Support Vector Regressor": SVR(),
    "KNN Regressor": KNeighborsRegressor()
}

results = []

# ===========================
# Train and Evaluate
# ===========================
for name, model in models.items():

    print("\n" + "="*60)
    print(name)

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    r2 = r2_score(y_test, y_pred)

    print("R2 Score :", r2)

    results.append([name, r2])

# ===========================
# Comparison Table
# ===========================
comparison = pd.DataFrame(
    results,
    columns=["Model", "R2 Score"]
)

comparison = comparison.sort_values(
    by="R2 Score",
    ascending=False
)

print("\n")
print("="*60)
print("MODEL COMPARISON")
print(comparison)