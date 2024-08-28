
#Data collection
import pandas as pd 
bostonhousing=pd.read_csv("BostonHousing.csv")

#Data exploration

#Split dataset to X and Y variables
#where X is the feature matrix and Y is the target Vector
Y=bostonhousing.medv
X=bostonhousing.drop(['medv'], axis=1)

#Data spilt into training and testing modules

from sklearn.model_selection import train_test_split
#size =0.2 means 80-20 ratio of training and testing
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)

#apply the Linear Regression Model

from sklearn import linear_model
from sklearn.metrics import mean_squared_error, r2_score

model = linear_model.LinearRegression()
model.fit(X_train, Y_train)

#making predictions
Y_pred = model.predict(X_test)

print('Coefficients:', model.coef_)
print('Intercept:', model.intercept_)
print('Mean squared error (MSE):',mean_squared_error(Y_test, Y_pred))

print('Coefficient of determination (R^2):',r2_score(Y_test, Y_pred))
print('Model Score:',model.score(X_test, Y_test))

#visualization by scatter plot

import seaborn as sns
import matplotlib.pyplot as plt

sns.scatterplot(x=Y_test, y=Y_pred)

plt.xlabel('Actual Prices (Y_test)')
plt.ylabel('Predicted Prices (Y_pred)')
plt.title('Actual vs Predicted Prices')

plt.show()