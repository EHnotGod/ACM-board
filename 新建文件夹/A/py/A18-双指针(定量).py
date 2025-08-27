# P1147 连续自然数和

m = int(input())
i = 1
j = 1
sum = 1
while i <= m // 2:
    if sum < m:
        j += 1
        sum += j
    if sum >= m:
        if sum == m:
            print(i, j)
        sum -= i
        i += 1