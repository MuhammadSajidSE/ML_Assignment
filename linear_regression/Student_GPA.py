import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
dataset = pd.read_csv('Students.csv')
x = dataset.iloc[:,:-1].values
y = dataset.iloc[:,1].values

from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test = train_test_split(x,y, test_size=1/4,random_state=42)

from sklearn.linear_model import LinearRegression

regressor = LinearRegression()
regressor.fit(x_train,y_train)

#trainning data show
plt.scatter(x_train,y_train,color='red')
plt.title('Study hour vs CGPA Training data')
plt.plot(x_train,regressor.predict(x_train),color='blue')
plt.xlabel('Study hour')
plt.ylabel('CGPA')
plt.show()

#testing data show
plt.scatter(x_test, y_test,color='red')
plt.title('Study hour vs CGPA testing data')
plt.plot(x_train,regressor.predict(x_train),color='blue')
plt.xlabel('Study hour')
plt.ylabel('CGPA')
plt.show()

CGPA_Predict = regressor.predict([[1714]])

R_square = regressor.score(x_train,y_train)