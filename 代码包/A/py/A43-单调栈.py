n = int(input())
a = list(map(int, input().split()))
f = [0] * n
q = []
for i in range(n - 1, -1, -1):
    while q and a[q[-1]] < a[i]:
        q.pop()
    if q:
        f[i] = q[-1]
    q.append(i)
print(*f)