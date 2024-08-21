import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split # 1111
from sklearn.linear_model import LinearRegression # 22222
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score

# (산점도,선그래프)
data = pd.read_excel('./data/2.elevator_failure_prediction.xlsx')
data_filled = data.fillna(0)
des=data.describe()
print(des)

array = data.values
X = array[:, 1:12]
Y = array[:, 12]
# 족립변수/종속변수

(X_train,X_text,
 Y_train,Y_test) = train_test_split(X,Y,test_size=0.2)
# 학습데이터/테스트 데이터
model = LinearRegression()
model.fit(X_train,Y_train)

y_pred = model.predict(X_text)

plt.scatter(range(len(X_text[:20])),Y_test[:20],color='red')
plt.scatter(range(len(X_text[:20])),y_pred[:20],color='blue')
plt.show()
mse = mean_squared_error(Y_test, y_pred)
print(mse)
# 학습모델 만들고 예측한거 한번만 한거
scaler = MinMaxScaler(feature_range=(0, 1))
rescaled_X = scaler.fit_transform(X)
kfold = KFold(n_splits=10, shuffle=True, random_state=40)
acc = cross_val_score(model, rescaled_X, Y, cv=kfold, scoring='neg_mean_squared_error')
print(-acc)
