import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB

from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# ===============================
# Load Dataset
# ===============================
df = pd.read_csv("heart.csv")

# Remove Duplicates
df = df.drop_duplicates()

# Convert Categorical Columns
df = pd.get_dummies(df, drop_first=True)

# Features and Target
X = df.drop("HeartDisease", axis=1)
y = df["HeartDisease"]

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

# ===============================
# Models
# ===============================
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "Support Vector Machine": SVC(),
    "KNN": KNeighborsClassifier(),
    "Naive Bayes": GaussianNB()
}

accuracy_list = []
for name, model in models.items():

    print("\n" + "="*60)
    print(name)

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    accuracy_list.append([name, accuracy])
    print("\nAccuracy Score")
    print(accuracy)
    print("\nConfusion Matrix")
    print(confusion_matrix(y_test, y_pred))
    print("\nClassification Report")
    print(classification_report(y_test, y_pred))

# ===============================
# Comparison Table
# ===============================
comparison = pd.DataFrame(
    accuracy_list,
    columns=["Model", "Accuracy"]
)

comparison = comparison.sort_values(
    by="Accuracy",
    ascending=False
)

print("\n")
print("="*60)
print("MODEL COMPARISON")
print(comparison)