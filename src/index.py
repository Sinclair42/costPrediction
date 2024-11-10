import numpy as np
import pandas as pd
import src.statistics.linear as linear

# 데이터를 불러오기
data = pd.read_csv("data.csv", encoding="cp949")

# 평균월세를 추출하여 시작 달을 0으로 변경(기준: 2020년 7월)
data = data.filter(regex="지역|평균월세가격")
data.set_index("지역", inplace=True)
data.columns = [i for i in range(len(data.columns))]
data_array = data.to_numpy()

# 서울, 경기도, 부산 추출
X = np.arange(len(data.columns))
seoul_data = data_array[3]
gyeongy_data = data_array[16]
busan_data = data_array[9]

# 서울 데이터 추측
print('---------------서울 데이터 추측----------------------')
seoul_coefficient = linear.Coefficient(0, 0)
seoul_linear = linear.LinearRegression(X, seoul_data)
seoul_result = seoul_linear.find_min()
print(f'학습 후 가중치, 편향: {seoul_result}')
print(f'초기 RSS: {seoul_linear.rss(seoul_coefficient)}')
print(f'학습 후 RSS: {seoul_linear.rss(seoul_result)}')
seoul_linear.draw_line(seoul_result)

# 경기 데이터 추측
print('---------------경기 데이터 추측----------------------')
gyeongy_coefficient = linear.Coefficient(0, 0)
gyeongy_linear = linear.LinearRegression(X, gyeongy_data)
gyeongy_result = gyeongy_linear.find_min()
print(f'학습 후 가중치, 편향: {gyeongy_result}')
print(f'초기 RSS: {gyeongy_linear.rss(gyeongy_coefficient)}')
print(f'학습 후 RSS: {gyeongy_linear.rss(gyeongy_result)}')
gyeongy_linear.draw_line(gyeongy_result)

# 부산 데이터 추측
print('---------------부산 데이터 추측----------------------')
busan_coefficient = linear.Coefficient(0, 0)
busan_linear = linear.LinearRegression(X, busan_data)
busan_result = busan_linear.find_min()
print(f'학습 후 가중치, 편향: {busan_result}')
print(f'초기 RSS: {busan_linear.rss(busan_coefficient)}')
print(f'학습 후 RSS: {busan_linear.rss(busan_result)}')
busan_linear.draw_line(busan_result)

# 2027년 3월의 월세 예측
delta_month = 12 * 7 - 4
for result in [seoul_result, gyeongy_result, busan_result]:
    print(result.predict(delta_month))
