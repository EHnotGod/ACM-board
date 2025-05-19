import sys
from collections import deque

def toposort():
    q = deque()
    for i in range(1, n + 1):
        if din[i] == 0:
            q.append(i)
    while q:
        x = q.popleft()
        tp.append(x)
        for y in e[x]:
            din[y] -= 1
            if din[y] == 0:
                q.append(y)
    return len(tp) == n

# 读取 n, m
n, m = map(int, sys.stdin.readline().split())
# 初始化邻接表和入度数组
e = [[] for _ in range(n + 1)]
din = [0] * (n + 1)
tp = []

# 读取边信息
for _ in range(m):
    a, b = map(int, sys.stdin.readline().split())
    e[a].append(b)
    din[b] += 1

# 执行拓扑排序并输出结果
if not toposort():
    print(-1)
else:
    print(*tp)