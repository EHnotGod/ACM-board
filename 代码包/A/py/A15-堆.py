# P3378 【模板】堆

import heapq

n = int(input())
q = []
while n:
    n -= 1
    op = list(map(int, input().split()))
    if op[0] == 1:
        heapq.heappush(q, op[1])
    elif op[0] == 2:
        print(q[0])
    else:
        heapq.heappop(q)