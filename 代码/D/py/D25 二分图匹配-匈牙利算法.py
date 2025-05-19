
import sys
sys.setrecursionlimit(1000000)

n, m, k = map(int, input().split())

N = 505
graph = [[] for _ in range(N)]
match = [0] * N
vis = [0] * N

for _ in range(k):
    a, b = map(int, input().split())
    graph[a].append(b)

def dfs(u):
    for v in graph[u]:
        if vis[v]:
            continue
        vis[v] = 1
        if match[v] == 0 or dfs(match[v]):
            match[v] = u
            return True
    return False

ans = 0
for u in range(1, n + 1):
    vis = [0] * N
    if dfs(u):
        ans += 1

print(ans)
