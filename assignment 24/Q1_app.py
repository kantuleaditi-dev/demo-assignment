import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
# Load Dataset
df = pd.read_csv("heart.csv")
# Display First 5 Rows
print("First 5 Rows")
print(df.head())

# Dataset Information
print("\nDataset Information")
print(df.info())
# Check Missing Values
print("\nMissing Values")
print(df.isnull().sum())
# Check Duplicate Rows
print("\nDuplicate Rows :", df.duplicated().sum())
# Remove Duplicate Rows
df = df.drop_duplicates()

# Convert Categorical Columns into Numeric
df = pd.get_dummies(df, drop_first=True)
X = df.drop("HeartDisease", axis=1)
y = df["HeartDisease"]
print("\nIndependent Features")
print(X.columns)
print("\nDependent Feature")
print(y.name)
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)
# Feature Scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
print("\nPreprocessing Completed Successfully")
print("X_train Shape :", X_train.shape)
print("X_test Shape :", X_test.shape)
print("y_train Shape :", y_train.shape)
print("y_test Shape :", y_test.shape)







import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Load Dataset
df = pd.read_csv("Housing.csv")

# Display First 5 Rows
print(df.head())

# Dataset Information
print(df.info())

# Missing Values
print(df.isnull().sum())

# Fill Missing Values
df["total_bedrooms"] = df["total_bedrooms"].fillna(df["total_bedrooms"].median())

# Remove Duplicates
df = df.drop_duplicates()

# Convert Categorical Column
df = pd.get_dummies(df, drop_first=True)

# Features and Target
X = df.drop("median_house_value", axis=1)
y = df["median_house_value"]

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Feature Scaling
scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

print("Preprocessing Completed Successfully")
print("X_train Shape:", X_train.shape)
print("X_test Shape:", X_test.shape)