from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# pandas 라이브러리를 불러옵니다
import pandas as pd

df = pd.read_csv('.\pima-indians-diabetes3.csv')

# 세부 정보를 X로 지정합니다.
X = df.iloc[:,0:8]
# 당뇨병 여부를 Y로 지정합니다.
y = df.iloc[:,8]

# 모델설정
model = Sequential()
# 레이어 추가 > 히든 노드 12개
model.add(Dense(12, input_dim=8, activation='relu', name='Dense_1'))
# x의 개수 (히든노드)
model.add(Dense(8, activation='relu', name='Dense_2'))
# 출력층 노드 1
model.add(Dense(1, activation='sigmoid',name='Dense_3'))
model.summary()

# 모델을 컴파일합니다.
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# 모델을 실행합니다.
history=model.fit(X, y, epochs=100, batch_size=5)
