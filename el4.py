import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

# 데이터 로딩
data = pd.read_excel('./data/2.elevator_failure_prediction.xlsx')

# 첫 번째 열 제외
data_features = data.iloc[:, 1:]  # 첫 번째 열 제외

# 데이터 타입 확인
print(data_features.dtypes)

# 날짜 데이터가 있는 경우, 이를 제외한 데이터만 추출
data_numeric = data_features.select_dtypes(include=[float, int])  # 수치형 데이터만 선택

# 마지막 열 (타겟 변수)
target = data_features.iloc[:, -1]

# 0인 값을 가진 행 제거 (마지막 열 포함)
data_cleaned = data_numeric[(data_numeric != 0).all(axis=1)]

# 데이터 스케일링
scaler = MinMaxScaler()
scaled_features = scaler.fit_transform(data_cleaned)
scaled_data = pd.DataFrame(scaled_features, columns=data_numeric.columns)

# 마지막 열 추가
scaled_data[data_features.columns[-1]] = target[data_cleaned.index].values

# 상관계수 행렬 계산 (스케일 조정된 데이터)
correlation_matrix = scaled_data.corr()

# 상관계수 행렬 출력
print(correlation_matrix)

# 상관관계 행렬 시각화
plt.figure(figsize=(12, 10))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1, fmt='.2f')
plt.title('Correlation Matrix with Min-Max Scaled Data (Excluding First Column)')
plt.show()
