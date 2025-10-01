# 평균 제곱 오차 확인해보기 // Mean Squared Error
import numpy as np

# 가상의 기울기 a와 y절편 b를 정합니다~
fake_a = 3
fake_b = 76

# 공부시간 x와 성적 y의 넘파이 배열을 만듦
x = np.array([2, 4, 6, 8])
y = np.array([81, 93, 91, 97])

# y=ax+b에 가상의 a값과 b값을 대입한 결과 출력해주는 함수
def predict(x):
    return fake_a * x + fake_b

# 예측값이 들어갈 빈 리스트
predict_result = []

# 모든 x값을 한번씩 대입해서 predict_result 리스트 만듦
for i in range(len(x)):
    predict_result.append(predict(x[i]))
    print(f"공부시간 = {x[i]}, 실제점수 = {y[i]}, 예측점수 = {predict(x[i])}")

# 평균 제곱 오차 함수를 각 y값에 대입해 최종 값을 구하는 함수
n = len(x)


def mse(y, y_pred):
    return (1/n) * sum((y-y_pred)**2)

# 평균 제곱 오차값을 출력합니다
print("평균 제곱 오차 : " + str(mse(y, predict_result)))