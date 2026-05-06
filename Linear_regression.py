# Import required libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

print("All libraries imported successfully!")



# Download and load boston housing dataset using kagglehub
import kagglehub
import os

path = kagglehub.dataset_download("vikrishnan/boston-house-prices")
print("Path to dataset files:", path)

# Load dataset
df = pd.read_csv(os.path.join(path, "housing.csv"),
                    header=None,
                    sep='\s+',
                    names=['CRIM','ZN','INDUS','CHAS','NOX','RM','AGE',
                    'DIS','RAD','TAX','PTRATIO','B','LSTAT','PRICE'])

print("\nDataset loaded successfully!")
print(f"Shape: {df.shape}")
print(df.head())











# Check missing values
print("=== Missing Values ===")
print(df.isnull().sum())

print("\n=== Any Missing Values? ===")
print(df.isnull().values.any())# Perform train test split
X = df.drop('price', axis=1)
y = df['price']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)





# Create model variable and train the model
model = LinearRegression()
model.fit(X_train, y_train)



# Perform prediction and calculate error
y_pred = model.predict(X_test)

print(f"MSE: {mean_squared_error(y_test, y_pred):.2f}")
print(f"R2 Score: {r2_score(y_test, y_pred):.2f}”)



# Plot Actual & Predicted rpice with linear regression line
plt.scatter(y_test, y_pred, alpha=0.7)
plt.plot([y.min(), y.max()], [y.min(), y.max()], color='red', linestyle='--')
plt.xlabel("Actual Prices")
plt.ylabel("Predicted Prices")
plt.title("Actual vs Predicted Prices")
plt.show()




sample_data = X_test.iloc[0:1]

predicted_price = model.predict(sample_data)

print("Predicted House Price:", predicted_price[0])
print("Actual House Price:", y_test.iloc[0])
