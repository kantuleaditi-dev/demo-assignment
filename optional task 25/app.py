import pandas as pd
import joblib

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, classification_report

# Load Iris Dataset
iris = load_iris()

X = pd.DataFrame(iris.data, columns=iris.feature_names)
y = iris.target

# Train Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# --------------------- SVM Grid Search ---------------------

param_grid = {
    "C": [0.1, 1, 10],
    "kernel": ["linear", "rbf"]
}

grid = GridSearchCV(
    estimator=SVC(),
    param_grid=param_grid,
    cv=5
)

grid.fit(X_train, y_train)

svm = grid.best_estimator_

svm_pred = svm.predict(X_test)

svm_acc = accuracy_score(y_test, svm_pred)

print("\nSVM Accuracy:", svm_acc)
print(classification_report(y_test, svm_pred))

# --------------------- Random Forest ---------------------

rf = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

rf.fit(X_train, y_train)

rf_pred = rf.predict(X_test)

rf_acc = accuracy_score(y_test, rf_pred)

print("\nRandom Forest Accuracy:", rf_acc)

# --------------------- AdaBoost ---------------------

ada = AdaBoostClassifier(
    n_estimators=100,
    random_state=42
)

ada.fit(X_train, y_train)

ada_pred = ada.predict(X_test)

ada_acc = accuracy_score(y_test, ada_pred)

print("\nAdaBoost Accuracy:", ada_acc)

# --------------------- Gradient Boosting ---------------------

gb = GradientBoostingClassifier(
    n_estimators=100,
    learning_rate=0.1,
    random_state=42
)

gb.fit(X_train, y_train)

gb_pred = gb.predict(X_test)

gb_acc = accuracy_score(y_test, gb_pred)

print("\nGradient Boosting Accuracy:", gb_acc)

# --------------------- Comparison ---------------------

result = pd.DataFrame({
    "Model": [
        "SVM",
        "Random Forest",
        "AdaBoost",
        "Gradient Boosting"
    ],
    "Accuracy": [
        svm_acc,
        rf_acc,
        ada_acc,
        gb_acc
    ]
})

print("\nModel Comparison")
print(result)

best_model = max(
    [
        (svm_acc, svm),
        (rf_acc, rf),
        (ada_acc, ada),
        (gb_acc, gb)
    ],
    key=lambda x: x[0]
)[1]

joblib.dump(best_model, "best_model.pkl")

print("\nBest model saved successfully as best_model.pkl")