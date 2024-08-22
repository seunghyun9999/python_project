import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score, confusion_matrix, mean_squared_error, mean_absolute_error

# 로지스틱 회귀를 모델을 만들어서 실행 한다음 종속변수를 선택한 확률을 보여주는 것

data = pd.read_excel('./data/2.elevator_failure_prediction.xlsx')
data.describe()
data_filled = data.fillna(0)
des = data_filled.describe()  # 전처리된 데이터의 통계량 확인
# print(des)

array = data_filled.values
X = array[:, 1:12]
Y = array[:, 12]

# Y 데이터에서 1과 2를 모두 2로 변경
Y = np.where(Y == 1, 2, Y).astype(int)

(X_train, X_test, Y_train, Y_test) = train_test_split(X, Y, test_size=0.2, random_state=42)

scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
# 모델 학습
model = LogisticRegression()
model.fit(X_train_scaled, Y_train)
# y_pred = model.predict_proba(X_test)
y_pred = model.predict(X_test_scaled)
print(y_pred)

# 정확도 계산
accuracy = accuracy_score(Y_test, y_pred)
print(f"Accuracy: {accuracy}")

# 오차 행렬 출력
conf_matrix = confusion_matrix(Y_test, y_pred)
print(f"Confusion Matrix:\n{conf_matrix}")

# K-Fold Cross Validation
kfold = KFold(n_splits=5, random_state=42, shuffle=True)
results = cross_val_score(model, X_train_scaled, Y_train, cv=kfold, scoring='accuracy')
print(f"K-Fold 교차검증 정확도: {results.mean()} ± {results.std()}")