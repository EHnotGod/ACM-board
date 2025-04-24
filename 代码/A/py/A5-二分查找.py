# P2249 查找

import bisect
n, m = map(int, input().split())
lis = list(map(int, input().split()))
q_lis = list(map(int, input().split()))
for i in range(m):
    q = q_lis[i]
    id = bisect.bisect_left(lis, q)
    if id >= n or lis[id] != q:
        print(-1, end=" ")
    else:
        print(id + 1, end=" ")