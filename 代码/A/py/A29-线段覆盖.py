# P1803

L = []
n = int(input())
for i in range(n):
    l, r = map(int, input().split())
    L.append([l, r])
L.sort(key=lambda x: x[1])
last = 0
cnt = 0
for i in range(n):
    if last <= L[i][0]:
        last = L[i][1]
        cnt += 1
print(cnt)