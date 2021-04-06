
### Load Libraries
"""
 
import pandas as pd
 
"""### Load Data"""
 
path = pd.read_csv('bangalore house price prediction OHE-data.csv')
 
path.head()
 
"""### Split Data"""
 
X = path.drop('price', axis=1)
y = path['price']
 
print('Shape of X = ', X.shape)
print('Shape of y = ', y.shape)
 
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=51)
 
print('Shape of X_train = ', X_train.shape)
print('Shape of y_train = ', y_train.shape)
print('Shape of X_test = ', X_test.shape)
print('Shape of y_test = ', y_test.shape)
 
"""### Feature Scaling"""
 
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
sc.fit(X_train)
X_train = sc.transform(X_train)
X_test = sc.transform(X_test)
 
"""## Linear Regression - ML Model Training"""
 
from sklearn.linear_model import LinearRegression
lr = LinearRegression()
 
lr.fit(X_train, y_train)
 
lr.coef_
 
lr.intercept_
 
"""## Predict the value of Home and Test"""
 
X_test[0, :]
 
lr.predict([X_test[0, :]])
 
lr.predict(X_test)
 
y_test
 
lr.score(X_test, y_test)
