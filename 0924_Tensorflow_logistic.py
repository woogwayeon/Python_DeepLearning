import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

x = np.array([2,4,6,8,10,12,14])
y = np.array([0,0,0,1,1,1,1])

model = Sequential() # 객체 만들때 조심하기
model.add(Dense(1, input_dim = 1, activation='sigmoid'))

model.compile(optimizer='sgd', loss='binary_crossentropy')
model.fit(x, y, epochs=2000)

plt.scatter(x, y)
plt.plot(x, model.predict(x), 'r')
plt.show()

hour = np.array([7])
prediction = model.predict([hour])

print(prediction)