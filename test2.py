from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv('./iris3.csv')

# 속성 X, 클래스 y
X = df.iloc[:,0:4]
y = df.iloc[:,4]

# 원-핫 인코딩 처리
# 범주형 데이터를 머신러닝 모델이 이해할 수 있는 수치형 데이터로 변환하는 기법
# 예를 들어 '사과', '배', '감'이 있다면 '과일_사과', '과일_배', '과일_감'과 같은 새로운 열을 만들고,
# '사과'인 데이터에는 '과일_사과' 열만 1로 표기
y = pd.get_dummies(y)

# 층(layer)이 순서대로 쌓이는 가장 기본적인 신경망 모델
model = Sequential()
# 레이어 추가 > 히든 노드 12개
model.add(Dense(12, input_dim=4, activation='relu'))
# x의 개수 (히든노드), relu : 0이하버림
model.add(Dense(8, activation='relu'))
# 출력층 노드, 3개의 출력값을 0~1 사이 확률로 바꿔줌
model.add(Dense(3, activation='softmax'))
model.summary()

# 종류가 여러개라서 카테고리컬 크로스엔트로피
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

history = model.fit(X, y, epochs=50, batch_size=5)

# 피마인디언 데이터엔 범주형데이터가 없음