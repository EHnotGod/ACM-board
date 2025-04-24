# http://poj.org/problem?id=2976

def check(x):
    s = 0
    for i in range(1, n + 1):
        c[i] = a[i] - x * b[i]
    c[1:n+1] = sorted(c[1:n+1])
    for i in range(k + 1, n + 1):
        s += c[i]
    return s >= 0
def find():
    l = 0
    r = 1
    while r - l > 1e-4:
        mid = (l + r) / 2
        if check(mid):
            l = mid
        else:
            r = mid
    return l


while 1:
    n, k = map(int, input().split())
    if n == 0 and k == 0:
        break
    a = list(map(int, input().split()))
    b = list(map(int, input().split()))
    a.insert(0, 0)
    b.insert(0, 0)
    c = [0] * (n + 1)
    print("{0:.0f}".format(100 * find()))
