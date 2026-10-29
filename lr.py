import numpy as np, matplotlib.pyplot as plt
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split

# 1. Load Data & Select AveRooms
housing = fetch_california_housing()
X = housing.data[:, housing.feature_names.index("AveRooms")].reshape(-1, 1)
y = housing.target
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)

# 2. Standardize Feature
mean, std = X_tr.mean(), X_tr.std()
X_tr_s, X_te_s = (X_tr - mean) / std, (X_te - mean) / std

# 3. Normal Equation (Fast Math Approach)
X_b = np.c_[np.ones((len(X_tr_s), 1)), X_tr_s]
theta = np.linalg.inv(X_b.T @ X_b) @ X_b.T @ y_tr

# 4. Gradient Descent Approach
w, b, lr, epochs, m = 0, 0, 0.01, 2000, len(y_tr)
for _ in range(epochs):
y_pred = w * X_tr_s.flatten() + b
w -= lr * (1/m) * np.sum((y_pred - y_tr) * X_tr_s.flatten())
b -= lr * (1/m) * np.sum(y_pred - y_tr)

# 5. Plotting (AveRooms)
X_sorted = np.sort(X_te, axis=0)
y_line = theta[0] + theta[1] * ((X_sorted - mean) / std)
plt.scatter(X_te, y_te, color='blue', alpha=0.5, label='Actual')
plt.plot(X_sorted, y_line, color='red', linewidth=2, label='Regression')
plt.xlabel('Average Number of Rooms (AveRooms)')
plt.ylabel('Median House Value')
plt.legend()
plt.show()
