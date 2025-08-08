
def gauss():
    c, r = 0, 0
    for c in range(n):
        t = r
        for i in range(r, n):
            if abs(a[i][c]) > abs(a[t][c]):
                t = i
        if abs(a[t][c]) < 1e-6:
            continue
        for i in range(c, n + 1):
            a[t][i], a[r][i] = a[r][i], a[t][i]
        for i in range(n, c - 1, -1):
            a[r][i] /= a[r][c]
        for i in range(r + 1, n):
            if abs(a[i][c]) > 1e-6:
                for j in range(n, c - 1, -1):
                    a[i][j] -= a[i][c] * a[r][j]
        r += 1
    if r < n:
        for i in range(r, n):
            if abs(a[i][n]) > 1e-6:
                return 2
        return 1
    for i in range(n - 1, -1, -1):
        for j in range(i + 1, n):
            a[i][n] -= a[i][j] * a[j][n]
    return 0

n = int(input())
a = []
for i in range(n):
    a.append(list(map(float, input().split())))
t = gauss()
if t:
    print("No Solution")
else:
    for i in range(n):
        print("{:.2f}".format(a[i][n]))