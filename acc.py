import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

# Load dataset
df = pd.read_csv("bank.csv", sep=';')

# Convert categorical data to numeric
df = pd.get_dummies(df, drop_first=True)

# Target column
target = [col for col in df.columns if col.startswith('y_')][0]

X = df.drop(target, axis=1)
y = df[target]

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train model
model = DecisionTreeClassifier()
model.fit(X_train, y_train)

# Prediction
y_pred = model.predict(X_test)

# Accuracy
accuracy = accuracy_score(y_test, y_pred)

print("Decision Tree Classifier")
print("Accuracy:", round(accuracy * 100, 2), "%")