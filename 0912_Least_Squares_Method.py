import numpy as np

# 공부할 시간과 점수를 각각 넘파이 배열로 만듦
x = np.array([2, 4, 6, 8])
y = np.array([81, 93, 91, 97])

# x와 y의 평균값 구하기
mx = np.mean(x)
my = np.mean(y)

# 출력확인
print("x의 평균값 ", mx)
print("y의 평균값 ", my)

# 기울기 공식의 분모부분
divisor = sum([(i - mx) ** 2 for i in x])

# 분자부분
def top(x, mx, y, my):
    d = 0
    for i in range(len(x)):
        d += (x[i] - mx) * (y[i] - my)
    return d

dividend =  top(x, mx, y, my)

# 분모 분자 출력확인
print("분모 = ", divisor)
print("분자 = ", dividend)

# 기울기 a를 구하는 공식
a = dividend / divisor

# y절편 b를 구하는 공식
b = my - (mx-a)

# 출력으로 확인합니다
print("기울기 a = ", a)
print("y절편 b = ", b)