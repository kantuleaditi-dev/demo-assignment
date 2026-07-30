import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsRegressor

# -------------------------------
# Classification Model (Heart)
# -------------------------------

heart = pd.read_csv("heart.csv")
heart = heart.drop_duplicates()

heart = pd.get_dummies(heart, drop_first=True)
X = heart.drop("HeartDisease", axis=1)
y = heart["HeartDisease"]
columns = X.columns
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42
)
heart_scaler = StandardScaler()

X_train = heart_scaler.fit_transform(X_train)
X_test = heart_scaler.transform(X_test)

best_classifier = LogisticRegression(max_iter=1000)
best_classifier.fit(X_train, y_train)

joblib.dump(best_classifier, "best_classification_model.pkl")
joblib.dump(heart_scaler, "heart_scaler.pkl")
joblib.dump(columns, "heart_columns.pkl")

print("Classification Model Saved Successfully")

# -------------------------------
# Regression Model (Housing)
# -------------------------------

house = pd.read_csv("Housing.csv")

house["total_bedrooms"] = house["total_bedrooms"].fillna(
    house["total_bedrooms"].median()
)

house = house.drop_duplicates()

house = pd.get_dummies(house, drop_first=True)

X = house.drop("median_house_value", axis=1)
y = house["median_house_value"]

columns = X.columns

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42
)

house_scaler = StandardScaler()

X_train = house_scaler.fit_transform(X_train)
X_test = house_scaler.transform(X_test)

best_regressor = KNeighborsRegressor()
best_regressor.fit(X_train, y_train)

joblib.dump(best_regressor, "best_regression_model.pkl")
joblib.dump(house_scaler, "house_scaler.pkl")
joblib.dump(columns, "house_columns.pkl")

print("Regression Model Saved Successfully")