# P1387 最大正方形

a = [[0 for i in range(103)] for j in range(103)]
b = [[0 for i in range(103)] for j in range(103)]
n, m = map(int, input().split())
for i in range(1, n + 1):
    temp = list(map(int, input().split()))
    for j in range(1, m + 1):
        a[i][j] = temp[j - 1]
        b[i][j] = b[i][j - 1] + b[i - 1][j] - b[i - 1][j - 1] + a[i][j]
ans = 1
for l in range(2, min(m, n) + 1):
    for i in range(l, n + 1):
        for j in range(l, m + 1):
            if b[i][j]-b[i-l][j]-b[i][j-l]+b[i-l][j-l] == l * l:
                ans = l
print(ans)