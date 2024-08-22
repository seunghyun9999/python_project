import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score


# 데이터 로딩 및 전처리
test = pd.read_csv('./data/test.csv')
train = pd.read_csv('./data/train.csv')

test_filled = test.fillna(0)
train_filled = train.fillna(0)

test_array = test_filled.values
test_X = test_array[:, 2:13]
test_Y = test_array[:, 13]
test_Y = np.where(test_Y == 1, 2, test_Y).astype(int)

train_array = train_filled.values
train_X = train_array[:, 2:13]
train_Y = train_array[:, 13]
Y = np.where(train_Y == 1, 2, train_Y).astype(int)


scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(train_X)
X_test_scaled = scaler.transform(test_X)

model = LogisticRegression()
model.fit(X_train_scaled, Y)
y_pred = model.predict(X_test_scaled)
print(y_pred)

accuracy = accuracy_score(test_Y, y_pred)
print(f"Accuracy: {accuracy}")