import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# 🔹 Step 1: Load dataset
data = pd.read_csv("train.csv")
print("Dataset loaded successfully ✅")
print("Shape:", data.shape)
print(data.head())

# 🔹 Step 2: Check target column
# In Kaggle's House Prices dataset, the target is 'SalePrice'
target_column = "SalePrice"

if target_column not in data.columns:
    raise ValueError(f"Target column '{target_column}' not found. Available columns: {data.columns}")

X = data.drop(columns=[target_column])
y = data[target_column]

# 🔹 Step 3: Handle categorical data (One-Hot Encoding for simplicity)
X = pd.get_dummies(X, drop_first=True)

# 🔹 Step 4: Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 🔹 Step 5: Train Model
model = LinearRegression()
model.fit(X_train, y_train)

# 🔹 Step 6: Evaluate
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print("Mean Squared Error:", mse)

# 🔹 Step 7: Example Prediction
print("Example predictions:", y_pred[:5])
