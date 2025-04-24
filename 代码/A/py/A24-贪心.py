# P1842

n = int(input())
a = []
for i in range(n):
    w, s = map(int, input().split())
    a.append((w, s, w + s))
a.sort(key=lambda x: x[2])
res = int(-2e9)
t = 0
for i in range(n):
    res = max(res, t - a[i][1])
    t += a[i][0]
print(res)
