f = [[0] * 20 for i in range(20)]
def init():
    for i in range(10):
        f[1][i] = 1
    for i in range(2, 20):
        for j in range(10):
            for k in range(10):
                if abs(j - k) >= 2:
                    f[i][j] += f[i - 1][k]
def dp(x):
    if x == 0:
        return 0
    xlis = []
    while x > 0:
        xlis.append(x % 10)
        x = x // 10
    m = len(xlis)
    res = 0
    last = -2
    for i in range(m - 1, -1, -1):
        new = xlis[i]
        if i == m - 1:
            for j in range(1, new):
                if abs(j - last) >= 2:
                    res += f[i + 1][j]
        else:
            for j in range(new):
                if abs(j - last) >= 2:
                    res += f[i + 1][j]
        if abs(new - last) < 2:
            break
        last = new
        if i == 0:
            res += 1
    for i in range(m - 1):
        for j in range(1, 10):
            res += f[i + 1][j]
    return res
init()
a, b = map(int, input().split())
print(dp(b) - dp(a - 1))
