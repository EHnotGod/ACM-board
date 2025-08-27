def floyd():
    for k in range(1, n + 1):
        for i in range(1, n + 1):
            for j in range(1, n + 1):
                d[i][j] = min(d[i][j], d[i][k] + d[k][j])


n, m = map(int, input().split())
d = [[int(1e20) for __ in range(n + 1)] for _ in range(n + 1)]
for i in range(1, n + 1):
    d[i][i] = 0
for i in range(m):
    a, b, c = map(int, input().split())
    d[a][b] = min(d[a][b], c)
    floyd()
for i in range(1, n + 1):
    print(*d[i][1:])
