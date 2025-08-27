# P1631 序列合并

import heapq
n = int(input())
q = []
a = list(map(int, input().split()))
a.insert(0, 0)
b = list(map(int, input().split()))
b.insert(0, 0)
id = [0] * (n + 1)
for i in range(1, n + 1):
    id[i] = 1
    heapq.heappush(q, [a[1] + b[i], i])
for j in range(n):
    print(q[0][0], end=" ")
    i = q[0][1]
    heapq.heappop(q)
    id[i] += 1
    heapq.heappush(q, [a[id[i]] + b[i], i])


