# P1908 逆序对

import sys
sys.setrecursionlimit(1000000)

def merge(l, r):
    global res, a, b
    if l >= r:
        return
    mid = (l + r) // 2
    merge(l, mid)
    merge(mid + 1, r)

    i = l; j = mid + 1; k = l
    while i <= mid and j <= r:
        if a[i] <= a[j]:
            b[k] = a[i]
            k += 1; i += 1
        else:
            b[k] = a[j]
            k += 1; j += 1
            res += mid - i + 1
    while i <= mid:
        b[k] = a[i]
        k += 1; i += 1
    while j <= r:
        b[k] = a[j]
        k += 1; j += 1
    for i in range(l, r + 1):
        a[i] = b[i]

n = int(input())
a = list(map(int, input().split()))
b = [0] * n
res = 0
merge(0, n - 1)
print(res)