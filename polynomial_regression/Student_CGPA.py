import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split

dataset = pd.read_csv('Students.csv')
x = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

x_train,x_test,y_train,y_test = train_test_split(x,y, test_size=1/3)


from sklearn.linear_model import LinearRegression
liner_regression = LinearRegression()
liner_regression.fit(x, y)


from sklearn.preprocessing import PolynomialFeatures

polynomial = PolynomialFeatures(degree=4)
x_poly = polynomial.fit_transform(x)
liner_regression2 = LinearRegression()
liner_regression2.fit(x_poly,y)

# visualizing the linear_regression
plt.scatter(x_train,y_train,color= 'red')
plt.plot(x_train,liner_regression.predict(x_train),color='blue')
plt.show()

#visualizing the polynomial regression
plt.scatter(x,y,color='red')
plt.plot(x,liner_regression2.predict(x_poly) ,color='blue')
plt.show()

x_grid = np.arange(min(x_train), max(x_train), 0.1)
x_grid = x_grid.reshape((len(x_grid), 1))  
plt.scatter(x_train,y_train,color= 'red')
plt.plot(x_grid, liner_regression2.predict(polynomial.fit_transform(x_grid)), color='blue')
plt.title('Polynomial Regression')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()

train_score = liner_regression.score(x_train, y_train)
test_score = liner_regression2.score(x_test, y_test)
