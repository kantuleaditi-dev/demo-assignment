# Q5 - Compare Logistic Regression, KNN and Naive Bayes
# Step 1: Import Libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Load Dataset
df = pd.read_csv("diabetes.csv")

# Features and Target
X = df.drop("Outcome", axis=1)
y = df["Outcome"]

# Train Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Feature Scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
# Logistic Regression
lr = LogisticRegression(random_state=42)
lr.fit(X_train, y_train)
lr_pred = lr.predict(X_test)
# KNN
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)
knn_pred = knn.predict(X_test)
# Naive Bayes
nb = GaussianNB()
nb.fit(X_train, y_train)
nb_pred = nb.predict(X_test)
# Comparison Table
comparison = pd.DataFrame({
    "Algorithm": ["Logistic Regression", "KNN", "Naive Bayes"],
    "Accuracy": [
        accuracy_score(y_test, lr_pred),
        accuracy_score(y_test, knn_pred),
        accuracy_score(y_test, nb_pred)
    ],
    "Precision": [
        precision_score(y_test, lr_pred, zero_division=0),
        precision_score(y_test, knn_pred, zero_division=0),
        precision_score(y_test, nb_pred, zero_division=0)
    ],
    "Recall": [
        recall_score(y_test, lr_pred, zero_division=0),
        recall_score(y_test, knn_pred, zero_division=0),
        recall_score(y_test, nb_pred, zero_division=0)
    ],
    "F1-Score": [
        f1_score(y_test, lr_pred, zero_division=0),
        f1_score(y_test, knn_pred, zero_division=0),
        f1_score(y_test, nb_pred, zero_division=0)
    ]
})

print("\nAlgorithm Comparison\n")
print(comparison)

best = comparison.loc[comparison["Accuracy"].idxmax()]

print("\nBest Algorithm:", best["Algorithm"])
print("Best Accuracy:", round(best["Accuracy"], 4))