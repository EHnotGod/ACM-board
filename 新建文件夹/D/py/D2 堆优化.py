import heapq

class Edge():
    def __init__(self, v, w):
        self.v = v
        self.w = w

def dijkstra(s):
    for i in range(n + 1):
        d[i] = int(1e20)
    d[s] = 0
    q = []
    heapq.heappush(q, [0, s])
    while len(q) > 0:
        t = heapq.heappop(q)
        u = t[1]
        if vis[u]:
            continue
        vis[u] = 1
        for ed in e[u]:
            v = ed.v
            w = ed.w
            if d[v] > d[u] + w:
                d[v] = d[u] + w
                heapq.heappush(q, [d[v], v])

n, m, s = map(int, input().split())
e = [[] for i in range(n + 1)]
d = [0] * (n + 1)
vis = [0] * (n + 1)
for i in range(m):
    a, b, c = map(int, input().split())
    e[a].append(Edge(b, c))

dijkstra(s)
for i in range(1, n + 1):
    print(d[i], end=" ")


