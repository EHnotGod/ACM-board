# P2440 木材加工

def check(x):
    y = 0
    for i in range(1, n + 1):
        y += a[i] // x
    return y >= k

def find():
    l = 0
    r = int(1e8 + 1)
    while l + 1 < r:
        mid = (l + r) // 2
        if check(mid):
            l = mid
        else:
            r = mid
    return l
n, k = map(int, input().split())
a = [int(input()) for _ in range(n)]
a.insert(0, 0)
print(find())