import sys
sys.setrecursionlimit(10**7)

def dfs(x):
    c[x] = -1
    for y in e[x]:
        if c[y] < 0:
            return False
        elif c[y] == 0:
            if not dfs(y):
                return False
    c[x] = 1
    tp.append(x)
    return True

def toposort():
    for x in range(1, n + 1):
        if c[x] == 0:
            if not dfs(x):
                return False
    tp.reverse()
    return True

# 读取 n, m
n, m = map(int, sys.stdin.readline().split())
# 初始化邻接表和染色数组
e = [[] for _ in range(n + 1)]
c = [0] * (n + 1)
tp = []

# 读取边信息
for _ in range(m):
    a, b = map(int, sys.stdin.readline().split())
    e[a].append(b)

# 执行拓扑排序并输出结果
if not toposort():
    print(-1)
else:
    print(*tp)
