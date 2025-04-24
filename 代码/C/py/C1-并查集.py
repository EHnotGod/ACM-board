# https://www.luogu.com.cn/problem/P3367

n, m = map(int, input().split())
pa = [i for i in range(n + 1)]
def find(x):
    if pa[x] == x:
        return x
    pa[x] = find(pa[x])
    return pa[x]
def union(x, y):
    pa[find(x)] = find(y)

for _ in range(m):
    z, x, y = map(int, input().split())
    if z == 1:
        union(x, y)
    else:
        if find(x) == find(y):
            print("Y")
        else:
            print("N")