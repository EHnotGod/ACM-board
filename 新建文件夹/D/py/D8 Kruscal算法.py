class Edge():
    def __init__(self, u, v, w):
        self.u = u
        self.v = v
        self.w = w

def find(x):
    if fa[x] == x:
        return x
    fa[x] = find(fa[x])
    return fa[x]
def union(x, y):
    fa[find(x)] = find(y)
def kruskal():
    global ans, cnt
    e.sort(key=lambda k:k.w)
    for i in range(m):
        x = find(e[i].u)
        y = find(e[i].v)
        if x != y:
            union(x, y)
            ans += e[i].w
            cnt += 1
    return cnt == n - 1


n, m = map(int, input().split())
fa = [i for i in range(n + 1)]
ans, cnt = 0, 0
e = []
for i in range(m):
    u, v, w = map(int, input().split())
    e.append(Edge(u, v, w))
if not kruskal():
    print("orz")
else:
    print(ans)
