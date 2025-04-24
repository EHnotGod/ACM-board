n, m = map(int, input().split())

v = [0] * 100005
w = [0] * 100005
f = [0] * 100005

cnt = 0
for _ in range(n):
    b, a, s = map(int, input().split())
    j = 1
    while j <= s:
        cnt += 1
        v[cnt] = j * a
        w[cnt] = j * b
        s -= j
        j <<= 1
    if s > 0:
        cnt += 1
        v[cnt] = s * a
        w[cnt] = s * b

for i in range(1, cnt + 1):
    for j in range(m, v[i] - 1, -1):
        f[j] = max(f[j], f[j - v[i]] + w[i])

print(f[m])