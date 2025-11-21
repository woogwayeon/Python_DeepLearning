from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.datasets import mnist         # 데이터셋
from tensorflow.keras.utils import to_categorical   # 더미와 비슷

import matplotlib.pyplot as plt
import numpy as np
from tensorflow.python.keras.saving.saved_model.load import metrics

# 트레인과 테스트를 나눔 // X_train(그림), y_train(숫자)
(X_train, y_train), (X_test, y_test) = mnist.load_data()
X_train = X_train.reshape(X_train.shape[0], 28, 28, 1).astype('float32') / 255
# 가로, 세로, 흑백값? 그리고 색의 농도때문에 255 곱해줌
X_test = X_test.reshape(X_test.shape[0], 28, 28, 1).astype('float32') / 255

y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

# 컨볼루션 신경망의 설정
model = Sequential()
# 네모모양에 커널사이즈 3*3, 그걸 32개
model.add(Conv2D(32, kernel_size=(3,3), input_shape=(28,28,1), activation='relu'))
model.add(Conv2D(64, (3,3), activation='relu'))
# 나온것중에 가장 큰 값만 적용하는거
model.add(MaxPooling2D(pool_size=(2,2)))
# 쫌쫌따리 죽여버림
model.add(Dropout(0.25))
# 일자로 쭉펴기
model.add(Flatten())

model.add(Dense(128, activation='relu'))
# 다시 랜덤하게 50% 죽임
model.add(Dropout(0.5))
# 0-1 사이로 해서
model.add(Dense(10, activation='softmax'))
model.summary()

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# modelpath = "./MNIST_CNN.hdf5" .keras 로 바꾸기
modelpath = "./MNIST_CNN.keras"

# verbose // 학습과정 보자
checkpointer = ModelCheckpoint(filepath=modelpath, monitor='val_loss', verbose=1, save_best_only=True)
early_stopping_callback = EarlyStopping(monitor='val_loss', patience=10) # 로스 10개 나오면 멈춰??

# 모델을 실행합니다
# validation_split=0.25 // 테스트셋 못쓰게해서 검증을.. 트레이닝데이터 일부를 사용함(훈련과정에서만)
# batch_size // 머신러닝 모델을 학습시킬 때 한 번에 처리하는 데이터의 개수
history = model.fit(X_train, y_train, validation_split=0.25, epochs=30, batch_size=200, verbose=0,
                    callbacks=[early_stopping_callback,checkpointer])

# 정확도 출력
print("\n Test Accuracy : %.4f" % (model.evaluate(X_test, y_test)[1]))

# 검증데이터 로스(오차)와 학습데이터 로스 저장
y_vloss = history.history['val_loss']
y_loss = history.history['loss']

# 그걸 그래프로 표현
x_len = np.arange(len(y_loss))
plt.plot(x_len, y_vloss, marker='.', c="red", label='Testset_loss')
plt.plot(x_len, y_loss, marker='.', c="green", label='Trainset_loss')

plt.legend(loc='upper right')
plt.grid()
plt.xlabel('epoch')
plt.ylabel('loss')
plt.show()





