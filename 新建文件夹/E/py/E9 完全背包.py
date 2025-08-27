N = 1010
n, m = map(int, input().split())
v = [0] * N
w = [0] * N
f = [0] * N
for i in range(n):
    v[i + 1], w[i + 1] = map(int, input().split())
for i in range(1, n + 1):
    for j in range(v[i], m + 1):
        f[j] = max(f[j], f[j - v[i]] + w[i])
print(f[m])
