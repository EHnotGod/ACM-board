import bisect
n = int(input())
b = [0] * (n + 10)
a = list(map(int, input().split()))
b[0] = int(-2e9)
len = 0
for i in range(n):
    if b[len] < a[i]:
        len += 1
        b[len] = a[i]
    else:
        b[bisect.bisect_left(b, a[i], 1, len)] = a[i]
print(len)
