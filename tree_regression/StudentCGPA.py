


# Decision Tree Regression

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import r2_score

# Importing the dataset
dataset = pd.read_csv('Students.csv')

# Define features (X) and target variable (y)
x = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

# Splitting the dataset into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=1/4, random_state=42)

# Linear Regression
regressor = LinearRegression()
regressor.fit(x_train, y_train)

# Visualizing the Linear Regression
plt.scatter(x_train, y_train, color='red')
plt.title('Study hour vs CGPA Training data')
plt.plot(x_train, regressor.predict(x_train), color='blue')
plt.xlabel('Study hour')
plt.ylabel('CGPA')
plt.show()

plt.scatter(x_test, y_test, color='red')
plt.title('Study hour vs CGPA Testing data')
plt.plot(x_train, regressor.predict(x_train), color='blue')
plt.xlabel('Study hour')
plt.ylabel('CGPA')
plt.show()

CGPA_Predict = regressor.predict([[9]])
R_square_linear = regressor.score(x_train, y_train)
print("Linear Regression R-squared:", R_square_linear)
print("Linear Regression Test R-squared:", regressor.score(x_test, y_test))

# Decision Tree Regression
regressor = DecisionTreeRegressor(random_state=0)
regressor.fit(x, y)

# Predicting a new result
y_pred = regressor.predict([[6.5]])
X_grid = np.arange(x.min(), x.max(), 0.01)
X_grid = X_grid.reshape((len(X_grid), 1))

plt.scatter(x, y, color='red')
plt.plot(X_grid, regressor.predict(X_grid), color='blue')
plt.title('Decision Tree Regression')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()

R_square_decision_tree = regressor.score(x, y)
print("Decision Tree Regression R-squared:", R_square_decision_tree)

# Polynomial Regression
polynomial = PolynomialFeatures(degree=4)
x_poly = polynomial.fit_transform(x)
x_train_poly = polynomial.fit_transform(x_train)
x_test_poly = polynomial.fit_transform(x_test)

liner_regression2 = LinearRegression()
liner_regression2.fit(x_poly, y)

# Visualizing the Polynomial Regression
plt.scatter(x_train, y_train, color='red')
plt.plot(x_train, liner_regression2.predict(x_train_poly), color='blue')
plt.show()

plt.scatter(x, y, color='red')
plt.plot(x, liner_regression2.predict(x_poly), color='blue')
plt.show()

x_grid = np.arange(x_train.min(), x_train.max(), 0.1)
x_grid = x_grid.reshape((len(x_grid), 1))
x_grid_poly = polynomial.fit_transform(x_grid)

plt.scatter(x_train, y_train, color='red')
plt.plot(x_grid, liner_regression2.predict(x_grid_poly), color='blue')
plt.title('Polynomial Regression')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()

R_square_polynomial = liner_regression2.score(x_poly, y)
print("Polynomial Regression R-squared:", R_square_polynomial)
print("Polynomial Regression Test R-squared:", liner_regression2.score(x_test_poly, y_test))
