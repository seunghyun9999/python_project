import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score

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

# 데이터 학습용과 검증용으로 분할
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42, stratify=y)

# 의사결정나무 모델 구성 및 학습
dt_model = DecisionTreeClassifier(class_weight=class_weights, random_state=42)
dt_model.fit(X_train, y_train)

# 예측 및 평가
y_pred = dt_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)

print("Accuracy:", accuracy)
print("Classification Report:\n", report)

