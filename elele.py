import pandas as pd
from sklearn.model_selection import train_test_split, KFold
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score
import numpy as np

# 데이터 로드
data = pd.read_excel('./data/2.elevator_failure_prediction.xlsx')

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

# 가중치 부여: Status 2에 더 큰 가중치를 부여
class_weights = {0: 1, 1: 2, 2: 5}

# K-Fold 교차 검증 설정
kf = KFold(n_splits=5, shuffle=True, random_state=42)  # 5-Fold 교차 검증

accuracy_scores = []
classification_reports = []

for train_index, test_index in kf.split(X_scaled):
    X_train, X_test = X_scaled[train_index], X_scaled[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]

    # 의사결정나무 모델 구성 및 학습
    dt_model = DecisionTreeClassifier(class_weight=class_weights, random_state=42)
    dt_model.fit(X_train, y_train)

    # 예측 및 평가
    y_pred = dt_model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    # classification_report에 zero_division=0을 추가하여 경고 방지
    report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)

    accuracy_scores.append(accuracy)
    classification_reports.append(report)

# 평균 정확도 계산
mean_accuracy = np.mean(accuracy_scores)
print("Mean Accuracy:", mean_accuracy)

# 평균 F1-Score 보고서 계산
average_report = {}
labels = list(classification_reports[0].keys())

for label in labels:
    if label != 'accuracy':  # 'accuracy'가 포함될 수 있으므로 제외
        avg_f1 = np.mean([report[label]['f1-score'] for report in classification_reports if label in report])
        average_report[label] = {'f1-score': avg_f1}

print("Average Classification Report:\n", average_report)
