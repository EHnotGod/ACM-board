# P1588 [USACO07OPEN] Catch That Cow S

def bfs():
    global x, y
    N = int(2e5 + 1)
    dis = [-1 for i in range(N)]
    dis[x] = 0
    l = 0
    q = []; q.append(x)
    while l < len(q):
        x = q[l]
        l += 1
        if x + 1 < N and dis[x + 1] == -1:
            dis[x + 1] = dis[x] + 1
            q.append(x + 1)
        if x - 1 > 0 and dis[x - 1] == -1:
            dis[x - 1] = dis[x] + 1
            q.append(x - 1)
        if 2 * x < N and dis[x * 2] == -1:
            dis[x * 2] = dis[x] + 1
            q.append(x * 2)
        if x == y:
            print(dis[y])
            return

t = int(input())
for i in range(t):
    x, y = map(int, input().split())
    bfs()
