
N = 5010
d = [0] * N
vis = [0] * N
class Edge():
    def __init__(self, v, w):
        self.v = v
        self.w = w
e = [[] for i in range(N)]

def prim(s):
    global ans, cnt
    for i in range(n + 1):
        d[i] = int(1e9)
    d[s] = 0
    for i in range(1, n + 1):
        u = 0
        for j in range(1, n + 1):
            if not vis[j] and d[j] < d[u]:
                u = j
        vis[u] = 1
        ans += d[u]
        if d[u] != 1e9:
            cnt += 1
        for ed in e[u]:
            v = ed.v
            w = ed.w
            if d[v] > w:
                d[v] = w
    return cnt == n
n, m = map(int, input().split())
ans, cnt = 0, 0
for i in range(m):
    a, b, c = map(int, input().split())
    e[a].append(Edge(b, c))
    e[b].append(Edge(a, c))
if not prim(1):
    print("orz")
else:
    print(ans)
