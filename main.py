import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# ---------------------------------------------------------
# 1. LOAD DATA (FROM LOCAL FILE)
# ---------------------------------------------------------
# 'churn_data.csv' must be in the same folder as this script
filename = 'churn_data.csv'

try :
    df = pd.read_csv(filename)
    print("-"*60)
    print(f" Success! Loaded {filename}. Shape {df.shape}")
except FileNotFoundError:
    print(f" Error Could not find {filename}. Make sure the CSV is in the same folder!")
    exit()

# ---------------------------------------------------------
# 2. DATA CLEANING
# ---------------------------------------------------------
# Force 'TotalCharges' to numeric, turning errors into NaN
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')

# Drop missing values and the CustomerID column
df.dropna(inplace=True)
df.drop(columns=['customerID'], inplace=True)

# ---------------------------------------------------------
# 3. PREPROCESSING
# ---------------------------------------------------------
# Convert Target 'Churn' to 1 and 0
# CORRECT (With colons)
df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0})

# One-Hot Encoding for categorical variables
df_dummies = pd.get_dummies(df, drop_first=True)

# ---------------------------------------------------------
# 4. SPLITTING & TRAINING
# ---------------------------------------------------------
X = df_dummies.drop('Churn', axis=1)
y = df_dummies['Churn']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("-"*30)

print("Training Random Forest Model... (This may take a second)")
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# ---------------------------------------------------------
# 5. EVALUATION
# ---------------------------------------------------------
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print("-"*60)
print(f"Model Accuracy: {accuracy * 100:.2f}%")
print("-"*60)
print("DETAILED REPORT:")
print(classification_report(y_test, y_pred))

# ---------------------------------------------------------
# 6. VISUALIZATION (THE EXTRA MILE)
# ---------------------------------------------------------
# This section creates a bar chart of the top 5 factors driving churn
feature_importances = pd.Series(model.feature_importances_, index=X.columns)
top_5 = feature_importances.sort_values(ascending=False).head(5)

plt.figure(figsize=(10, 6))
sns.barplot(x=top_5.values, y=top_5.index, hue=top_5.index, palette='viridis', legend=False)
plt.title('Top 5 Factors Driving Customer Churn')
plt.xlabel('Importance Score')
plt.ylabel('Feature')
plt.show()  # This will open a window with the graph