import numpy as np
import matplotlib.pyplot as plt

# 선형회귀랑 범주형(클래스, 레이블 유의어) : 로지스틱회귀 // pip install tensorflow
# 선형회귀를 시그모이드 함수에 넣으면 로지스틱회귀 (0, 1)
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

x = np.array([[2, 0], [4, 4], [6, 2], [8, 3]])
y = np.array([81, 93, 91, 97])

model = Sequential()
model.add(Dense(1, input_dim=2, activation='linear'))
model.compile(optimizer='sgd', loss='mse') # 선형회귀 - mse(평균제곱오차)
model.fit(x, y, epochs = 2000)             # 로지스틱회귀 - CEE(cross entropy error)

hour = np.array([7])
private_class = np.array([4])

prediction = model.predict(np.array([[hour, private_class]]))

print("공부시간 = %.f  과외시간 = %.f  예상점수 = %.02f" % (hour, private_class, prediction))