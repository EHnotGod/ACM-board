import sys
import math
from collections import defaultdict
sys.setrecursionlimit(10**7)
input = sys.stdin.readline
n, m, k = map(int, input().split())
a = [0] + list(map(int, input().split()))

B = int(math.sqrt(n))

q = []
for i in range(m):
    l, r = map(int, input().split())
    q.append((l, r, i))

def mo_key(x):
    block = x[0] // B
    return (block, x[1] if block % 2 == 0 else -x[1])

q.sort(key=mo_key)

c = defaultdict(int)
sum = 0

def add(x):
    global sum
    sum -= c[x] * c[x]
    c[x] += 1
    sum += c[x] * c[x]

def remove(x):
    global sum
    sum -= c[x] * c[x]
    c[x] -= 1
    sum += c[x] * c[x]

ans = [0] * m
l, r = 1, 0
for L, R, idx in q:
    while l > L:
        l -= 1
        add(a[l])
    while r < R:
        r += 1
        add(a[r])
    while l < L:
        remove(a[l])
        l += 1
    while r > R:
        remove(a[r])
        r -= 1
    ans[idx] = sum

print("\n".join(map(str, ans)))