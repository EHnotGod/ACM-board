# P1923 【深基9.例4】求第 k 小的数

import sys
sys.setrecursionlimit(5000000)
def qnth_element(l, r):
    global a
    if l == r:
        return a[l]
    i = l - 1
    j = r + 1
    x = a[(l + r) // 2]
    while i < j:
        while 1:
            i += 1
            if not a[i] < x:
                break
        while 1:
            j -= 1
            if not a[j] > x:
                break
        if i < j:
            a[i], a[j] = a[j], a[i]
    if k <= j:
        return qnth_element(l, j)
    else:
        return qnth_element(j + 1, r)
n, k = map(int, input().split())
a = list(map(int, input().split()))
print(qnth_element(0, n - 1))