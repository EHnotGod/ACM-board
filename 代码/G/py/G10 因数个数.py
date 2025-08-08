n = int(input())
p = [0] * (n + 1)
vis = [0] * (n + 1)
a = [0] * (n + 1)
d = [0] * (n + 1)
cnt = 0
d[1] = 1
for i in range(2, n + 1):
    if not vis[i]:
        cnt += 1; p[cnt] = i
        a[i] = 1; d[i] = 2
    for j in range(1, n + 1):
        if i * p[j] > n:
            break
        m = i * p[j]
        vis[m] = 1
        if i % p[j] == 0:
            a[m] = a[i] + 1
            d[m] = d[i] // a[m] * (a[m] + 1)
            break
        else:
            a[m] = 1
            d[m] = d[i] * 2
for i in range(1, n + 1):
    print(d[i], end=" ")