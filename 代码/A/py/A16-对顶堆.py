# 对应题目：https://www.luogu.com.cn/problem/P7072

import heapq
a = [] # 大根堆
b = [] # 小根堆
n, w = map(int, input().split())
lis = list(map(int, input().split()))
for i in range(n):
    x = lis[i]
    k = max(1, int((i + 1) * w / 100))
    if len(b) == 0 or x >= b[0]:
        heapq.heappush(b, x)
    else:
        heapq.heappush(a, -x)
    while len(b) > k:
        heapq.heappush(a, -heapq.heappop(b))
    while len(b) < k:
        heapq.heappush(b, -heapq.heappop(a))
    print(b[0], end=" ")