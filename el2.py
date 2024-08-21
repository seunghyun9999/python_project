import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score

# 데이터 로딩 및 전처리
data = pd.read_excel('./data/2.elevator_failure_prediction.xlsx')
data_filled = data.fillna(0)  # 결측값 처리
des = data_filled.describe()  # 전처리된 데이터의 통계량 확인
print(des)

# 데이터 변환
array = data_filled.values
X = array[:, 1:12]
Y = array[:, 12]

# 학습 데이터와 테스트 데이터 분리
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

# 데이터 스케일링
scaler = MinMaxScaler(feature_range=(0, 1))
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 모델 학습
model = LinearRegression()
model.fit(X_train_scaled, Y_train)

# 예측
y_pred = model.predict(X_test_scaled)

# 시각화
plt.scatter(range(len(Y_test[:20])), Y_test[:20], color='red', label='True values')
plt.scatter(range(len(y_pred[:20])), y_pred[:20], color='blue', label='Predicted values')
plt.legend()
plt.show()

# 모델 평가
mse = mean_squared_error(Y_test, y_pred)
print(f'Mean Squared Error: {mse}')

# 교차 검증
rescaled_X = scaler.fit_transform(X)  # 전체 데이터 스케일링
kfold = KFold(n_splits=10, shuffle=True, random_state=40)
acc = cross_val_score(model, rescaled_X, Y, cv=kfold, scoring='neg_mean_squared_error')
print(f'Cross-validated MSE: {-np.mean(acc)}')
