import sys
sys.setrecursionlimit(100000)
input = sys.stdin.readline
n = int(input())
e = [[] for i in range(n + 1)]
for i in range(n - 1):
    u, v = map(int, input().split())
    e[u].append(v); e[v].append(u)
siz = [0] * (n + 1)
f = [0] * (n + 1)
cnt = int(1e9)
def dfs(u, p):
    global cnt
    siz[u] = 1
    for v in e[u]:
        if v != p:
            dfs(v, u)
            siz[u] += siz[v]
            f[u] = max(f[u], siz[v])
    f[u] = max(f[u], n - siz[u])
    cnt = min(cnt, f[u])
dfs(1, 0)
for i in range(1, n + 1):
    if f[i] == cnt:
        print(i, end = " ")