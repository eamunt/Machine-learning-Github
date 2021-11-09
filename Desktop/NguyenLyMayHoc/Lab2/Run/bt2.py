import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error
import numpy as np
dulieu = pd.read_csv("housing_RT.csv", index_col=0)
dulieu.iloc[1:5,] #lấy hàng từ 1 đến 5
X_train,X_test,y_train,y_test = train_test_split(dulieu.iloc[:,1:5],dulieu.iloc[:,0], test_size=1/3.0, random_state=100)
X_train.iloc[1:5,]
X_test[1:5]
y_test[1:5]

regressor = DecisionTreeRegressor(random_state = 0)
regressor.fit(X_train, y_train)

y_pred = regressor.predict(X_test)

err = mean_squared_error(y_test, y_pred)
print(err)

print(np.sqrt(err))
