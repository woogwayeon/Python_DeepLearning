from keras.src.callbacks import ModelCheckpoint
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
# 머신러닝 모델 학습 시 검증 손실(또는 정확도)이 더 이상 개선되지 않을 때 학습을 조기 중단하는 기법
# 이를 통해 과적합(overfitting)을 방지하고, 최적의 모델이 완성되는 시점에 학습을 멈추는 것
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt
import seaborn as sns

import pandas as pd
import numpy as np
from tensorflow.python.profiler.profiler_client import monitor

from cnntest import early_stopping_callback, checkpointer

df = pd.read_csv("./house_train.csv")

# 카테고리형 변수를 0과 1로 이루어진 변수로 변경
df = pd.get_dummies(df)

# 결측치를 전체 컬럼의 평균으로 대체해서 채워줌
df = df.fillna(df.mean())

# 상관관계 분석
df_corr = df.corr()

df_corr_sort = df_corr.sort_values('SalePrice', ascending=False)

cols_train=['OverallQual', 'GrLivArea', 'GarageCars', 'GarageArea', 'TotalBsmtSF']
X_train_pre = df[cols_train]

y = df['SalePrice'].values

# 전체의 80%를 학습셋으로, 20%는 테스트셋으로 설정
X_train, X_test, y_train, y_test = train_test_split(X_train_pre, y, test_size=0.2)

model = Sequential()
model.add(Dense(10, input_dim=X_train.shape[1], activation='relu'))
model.add(Dense(30, activation='relu'))
model.add(Dense(40, activation='relu'))
model.add(Dense(1))
model.summary()

# 평균제곱오차(연속데이터라서 이거쓰나..)
model.compile(optimizer='adam', loss='mean_squared_error')

# 20번 이상 결과가 향상되지 않으면 자동으로 중단 설정
early_stopping_callback = EarlyStopping(monitor='val_loss', patience=20)

modelpath = './ch15-house.keras'

checkpointer = ModelCheckpoint(filepath=modelpath, monitor='val_loss', verbose=1, save_best_only=True)

# 실행 환경설정 > 전체의 20%를 검증셋으로 설정
# batch_size // 머신러닝 모델을 학습시킬 때 한 번에 처리하는 데이터의 개수
# epochs // 머신러닝 모델을 학습시킬 때 전체 훈련 데이터셋을 한 번 모두 학습하는 횟수
history = model.fit(X_train, y_train, validation_split=0.25, epochs=2000, batch_size=32,
                    callbacks=[early_stopping_callback, checkpointer])

model.save('./house-model.keras')
real_prices = []
pred_prices = []
X_num = []

n_iter = 0
Y_predication = model.predict(X_test).flatten()

for i in range(25):
    real = y_test[i]
    prediction = Y_predication[i]
    print("실제가격 : {:.2f}, 예상가격 : {:.2f}".format(real, prediction))
    real_prices.append(real)
    pred_prices.append(prediction)
    n_iter = n_iter+1
    X_num.append(n_iter)

plt.plot(X_num, pred_prices, label='predicted price')
plt.plot(X_num, real_prices, label='real price')
plt.legend()
plt.show()