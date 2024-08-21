import pandas as pd

# 데이터 로드
data = pd.read_excel('./data/2.elevator_failure_prediction.xlsx')

# 1열(첫 번째 열) 제거
data = data.iloc[:, 1:]

# 온도, 진동, 센서1, 센서2, 센서5 중 하나라도 0인 경우 제거
filtered_data = data[~((data['Temperature'] == 0) &
                       (data['Vibrations'] == 0) &
                       (data['Sensor1'] == 0) &
                       (data['Sensor2'] == 0) &
                       (data['Sensor5'] == 0) &
                       (data['Status'] == 2))]

# 가중치 정의: Status 값에 따라 가중치 부여
weights = filtered_data['Status'].map({0: 1, 1: 2, 2: 3})

# 가중치를 반영한 개수 계산
count_0 = (weights[filtered_data['Status'] == 0]).sum()
count_2 = (weights[filtered_data['Status'] == 2]).sum()

# 결과 출력
print(f"Status가 0인 데이터의 가중치 합계: {count_0}")
print(f"Status가 2인 데이터의 가중치 합계: {count_2}")