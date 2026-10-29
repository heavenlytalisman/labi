import numpy as np, pandas as pd, matplotlib.pyplot as plt, seaborn as sns
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

# 1. Load Dataset & Select Displacement
data = sns.load_dataset("mpg").dropna()
X, y = data[['displacement']], data['mpg']
plt.scatter(X, y, color='blue', alpha=0.5, label='Actual Data')
X_seq = pd.DataFrame({'displacement': np.linspace(X.min()[0], X.max()[0], 300)})

# 2. Linear Regression Line
lin_model = LinearRegression().fit(X, y)
plt.plot(X_seq, lin_model.predict(X_seq), color='black', label='Linear')

# 3. Polynomial Regression Lines (Degrees 2, 3, 4)
colors = ['red', 'green', 'orange']
for d, c in zip([2, 3, 4], colors):
poly = PolynomialFeatures(degree=d)
model = LinearRegression().fit(poly.fit_transform(X), y)
plt.plot(X_seq, model.predict(poly.transform(X_seq)), color=c, label=f'Degree {d}')
plt.xlabel('Engine Displacement')
plt.ylabel('Miles Per Gallon (MPG)')
plt.title('Polynomial Regression vs Linear Regression')
plt.legend()
plt.show()
