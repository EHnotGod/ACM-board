import sys
sys.setrecursionlimit(int(1e7))
N = 6010
head = [0] * N
to = [0] * N
ne = [0] * N
idx = 0
def add(a, b):
    global idx
    idx += 1
    to[idx] = b; ne[idx] = head[a]; head[a] = idx
w = [0] * N
fa = [0] * N
f = [[0 for i in range(2)] for j in range(N)]
def dfs(u):
    f[u][1] = w[u]
    i = head[u]
    while i != 0:
        v = to[i]
        dfs(v)
        f[u][0] += max(f[v][0], f[v][1])
        f[u][1] += f[v][0]
        i = ne[i]
n = int(input())
for i in range(1, n + 1):
    w[i] = int(input())
for i in range(n - 1):
    a, b = map(int, input().split())
    add(b, a)
    fa[a] = True
root = 1
while fa[root]:
    root += 1
dfs(root)
print(max(f[root][0], f[root][1]))