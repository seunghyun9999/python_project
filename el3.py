import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# 데이터 로딩 및 전처리
data = pd.read_excel('./data/2.elevator_failure_prediction.xlsx')
data_filled = data.fillna(0)  # 결측값 처리

# 상관계수 행렬 계산
correlation_matrix = data_filled.corr()

# 상관계수 행렬 출력
print(correlation_matrix)

# 상관관계 행렬 시각화
plt.figure(figsize=(12, 10))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1, fmt='.2f')
plt.title('Correlation Matrix')
plt.show()
