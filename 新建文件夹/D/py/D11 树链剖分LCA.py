import sys
sys.setrecursionlimit(int(1e7))
N = 500010
fa = [0] * N
son = [0] * N
dep = [0] * N
siz = [0] * N
top = [0] * N
e = [[] for i in range(N)]
def dfs1(u, f):
    fa[u] = f
    siz[u] = 1
    dep[u] = dep[f] + 1
    for v in e[u]:
        if v == f:
            continue
        dfs1(v, u)
        siz[u] += siz[v]
        if siz[son[u]] < siz[v]:
            son[u] = v
def dfs2(u, t):
    top[u] = t
    if not son[u]:
        return
    dfs2(son[u], t)
    for v in e[u]:
        if v == fa[u] or v == son[u]:
            continue
        dfs2(v, v)
def lca(u, v):
    while top[u] != top[v]:
        if dep[top[u]] < dep[top[v]]:
            u, v = v, u
        u = fa[top[u]]
    if dep[u] < dep[v]:
        return u
    else:
        return v
n, m, s = map(int, input().split())
for i in range(1, n):
    a, b = map(int, input().split())
    e[a].append(b)
    e[b].append(a)
dfs1(s, 0)
dfs2(s, s)
for i in range(m):
    a, b = map(int, input().split())
    print(lca(a, b))