import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.metrics import r2_score, mean_squared_error
import joblib

DATAFILE = "house_prices.csv"

if os.path.exists(DATAFILE):
    data = pd.read_csv(DATAFILE)
else:
    np.random.seed(42)
    n = 500
    area = np.random.normal(1200, 300, n).clip(200, 5000)
    bedrooms = np.random.choice([1,2,3,4,5], size=n, p=[0.05,0.2,0.45,0.2,0.1])
    bathrooms = np.clip(np.round(bedrooms - np.random.choice([0,0.5,1], n, p=[0.6,0.3,0.1])), 1, 4)
    age = np.random.randint(0, 40, n)
    price = (50 * area) + (15000 * bedrooms) + (10000 * bathrooms) - (800 * age) + 0.05 * (area ** 1.5) + np.random.normal(0, 40000, n)
    data = pd.DataFrame({
        "Area": area.round(0).astype(int),
        "Bedrooms": bedrooms,
        "Bathrooms": bathrooms,
        "Age": age,
        "Price": price.round(2)
    })

num_cols = data.select_dtypes(include=[np.number]).columns.tolist()
num_cols.remove("Price")

for c in num_cols:
    if data[c].isnull().any():
        data[c].fillna(data[c].median(), inplace=True)

plt.figure(figsize=(6,5))
sns.heatmap(data.corr(), annot=True, fmt=".2f", cmap="coolwarm")
plt.show()

X = data[["Area", "Bedrooms", "Bathrooms", "Age"]]
y = data["Price"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

pipe_lr = Pipeline([
    ("scaler", StandardScaler()),
    ("lr", LinearRegression())
])

pipe_lr.fit(X_train, y_train)
y_pred_lr = pipe_lr.predict(X_test)

def metrics(y_true, y_pred):
    r2 = r2_score(y_true, y_pred)
    rmse = mean_squared_error(y_true, y_pred, squared=False)
    return r2, rmse

r2_lr, rmse_lr = metrics(y_test, y_pred_lr)

pipe_ridge = Pipeline([
    ("scaler", StandardScaler()),
    ("ridge", Ridge())
])

param_grid_ridge = {"ridge__alpha": [0.01, 0.1, 1.0, 10.0, 50.0, 100.0]}
grid_ridge = GridSearchCV(pipe_ridge, param_grid_ridge, cv=5, scoring="neg_root_mean_squared_error", n_jobs=-1)
grid_ridge.fit(X_train, y_train)

best_ridge = grid_ridge.best_estimator_
y_pred_ridge = best_ridge.predict(X_test)
r2_ridge, rmse_ridge = metrics(y_test, y_pred_ridge)

pipe_poly = Pipeline([
    ("scaler", StandardScaler()),
    ("poly", PolynomialFeatures(include_bias=False)),
    ("lr", LinearRegression())
])

param_grid_poly = {"poly__degree": [2, 3]}
grid_poly = GridSearchCV(pipe_poly, param_grid_poly, cv=5, scoring="r2", n_jobs=-1)
grid_poly.fit(X_train, y_train)

best_poly = grid_poly.best_estimator_
y_pred_poly = best_poly.predict(X_test)
r2_poly, rmse_poly = metrics(y_test, y_pred_poly)

pipe_poly_ridge = Pipeline([
    ("scaler", StandardScaler()),
    ("poly", PolynomialFeatures(include_bias=False)),
    ("ridge", Ridge())
])

param_grid_poly_ridge = {
    "poly__degree": [2, 3],
    "ridge__alpha": [0.01, 0.1, 1, 10]
}

grid_poly_ridge = GridSearchCV(pipe_poly_ridge, param_grid_poly_ridge, cv=5, scoring="r2", n_jobs=-1)
grid_poly_ridge.fit(X_train, y_train)

best_poly_ridge = grid_poly_ridge.best_estimator_
y_pred_poly_ridge = best_poly_ridge.predict(X_test)
r2_poly_ridge, rmse_poly_ridge = metrics(y_test, y_pred_poly_ridge)

results = pd.DataFrame({
    "Model": ["Linear", "Ridge", "Polynomial", "Polynomial+Ridge"],
    "R2": [r2_lr, r2_ridge, r2_poly, r2_poly_ridge],
    "RMSE": [rmse_lr, rmse_ridge, rmse_poly, rmse_poly_ridge]
})

print(results)

plt.figure(figsize=(8,6))
plt.scatter(y_test, y_pred_lr, alpha=0.6, label="Linear")
plt.scatter(y_test, y_pred_ridge, alpha=0.6, label="Ridge")
plt.scatter(y_test, y_pred_poly_ridge, alpha=0.6, label="Poly+Ridge")
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], "k--")
plt.xlabel("Actual Price")
plt.ylabel("Predicted Price")
plt.legend()
plt.show()

joblib.dump(best_poly_ridge, "best_model_joblib.pkl")

new_house = pd.DataFrame({
    "Area": [1500, 2500],
    "Bedrooms": [3, 4],
    "Bathrooms": [2, 3],
    "Age": [5, 10]
})

preds = best_poly_ridge.predict(new_house)
print(preds)