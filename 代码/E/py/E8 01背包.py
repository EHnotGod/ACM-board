N = 3410
M = 13000
n, m = map(int, input().split())
v = [0] * N
w = [0] * N
f = [0] * M
for i in range(n):
    v[i + 1], w[i + 1] = map(int, input().split())
for i in range(1, n + 1):
    for j in range(m, 0, -1):
        if j >= v[i]:
            f[j] = max(f[j], f[j - v[i]] + w[i])
        else:
            break
print(f[m])
