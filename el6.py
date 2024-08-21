import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.model_selection import KFold, cross_val_score, cross_val_predict
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np  # numpy 모듈 import

# 데이터 로드
data = pd.read_excel('./data/2.elevator_failure_prediction.xlsx')

# 결측값을 이전 값으로 채우기
data = data.ffill()

# 1열(첫 번째 열) 제거
data = data.iloc[:, 1:]

# 온도, 진동, 센서1, 센서2, 센서5이 0인 경우를 제거
filtered_data = data[~((data['Temperature'] == 0) &
                       (data['Vibrations'] == 0) &
                       (data['Sensor1'] == 0) &
                       (data['Sensor2'] == 0) &
                       (data['Sensor5'] == 0) &
                       (data['Status'] == 2))]

# 독립변수와 종속변수 분리
X = filtered_data.drop('Status', axis=1)
y = filtered_data['Status']

# 데이터 정규화
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 의사결정나무 모델 구성
dt_model = DecisionTreeClassifier(class_weight={0: 1, 1: 2, 2: 5}, random_state=42)

# KFold 설정
kf = KFold(n_splits=5, shuffle=True, random_state=42)

# K-Fold Cross Validation을 사용하여 정확도 평가
scores = cross_val_score(dt_model, X_scaled, y, cv=kf, scoring='accuracy')
y_pred_kfold = cross_val_predict(dt_model, X_scaled, y, cv=kf)

# 평균 정확도 및 표준편차 출력
print(f"Average Accuracy: {scores.mean():.4f}")
print(f"Standard Deviation of Accuracy: {scores.std():.4f}")

# 최종 리포트 출력 (K-Fold 전체에 대한 예측 결과)
print("\nClassification Report:\n", classification_report(y, y_pred_kfold))

# 혼동 행렬 시각화
conf_matrix = confusion_matrix(y, y_pred_kfold)
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=['Class 0', 'Class 1', 'Class 2'], yticklabels=['Class 0', 'Class 1', 'Class 2'])
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.show()

# 특성 중요도 시각화
importances = dt_model.fit(X_scaled, y).feature_importances_
indices = np.argsort(importances)[::-1]
features = X.columns

plt.figure(figsize=(10, 6))
plt.title('Feature Importances')
plt.bar(range(X.shape[1]), importances[indices], align='center')
plt.xticks(range(X.shape[1]), features[indices], rotation=90)
plt.xlim([-1, X.shape[1]])
plt.show()

