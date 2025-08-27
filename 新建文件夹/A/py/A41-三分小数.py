import sys
import math

INF = float('inf')
EPS = 1e-9

def f(x, n, a, b, c):
    maxn = -INF
    for i in range(n):
        val = a[i] * x * x + b[i] * x + c[i]
        maxn = max(maxn, val)
    return maxn

t = int(input())
for _ in range(t):
    n = int(input())
    a = []
    b = []
    c = []
    for _ in range(n):
        ai, bi, ci = map(int, input().split())
        a.append(ai)
        b.append(bi)
        c.append(ci)

    l = 0.0
    r = 1000.0
    while r - l > EPS:
        m1 = (2 * l + r) / 3
        m2 = (l + 2 * r) / 3
        if f(m1, n, a, b, c) < f(m2, n, a, b, c):
            r = m2
        else:
            l = m1
    print(f"{f(l, n, a, b, c):.4f}")
