import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv('test.csv')
x = data.iloc[:,:-1]
y = data.iloc[:,1]

from sklearn.model_selection import train_test_split
x_train, x_test,y_train,y_test = train_test_split(x,y, test_size=1/4,random_state=0)

from sklearn.linear_model import LinearRegression
regresser = LinearRegression()
regresser.fit(x_train,y_train)

y_predict = regresser.predict(x_test)

plt.scatter(x_train, y_train, color='red',s=5)
plt.plot(x_train, regresser.predict(x_train),color='blue')
plt.title('Train data')
plt.show()

plt.scatter(x_test, y_test, color='red',s=5)
plt.plot(x_train, regresser.predict(x_train),color='blue')
plt.title('Test data')
plt.show()

R_square = regresser.score(x_train, y_train)