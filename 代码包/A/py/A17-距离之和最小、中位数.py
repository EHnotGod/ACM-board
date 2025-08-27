# CF1486B Eastern Exhibition

t = int(input())
for _ in range(t):
    n = int(input())
    a = [0] * n
    b = [0] * n
    for i in range(n):
        a[i], b[i] = map(int, input().split())
    a.sort()
    b.sort()
    x = a[n // 2] - a[(n - 1) // 2] + 1
    y = b[n // 2] - b[(n - 1) // 2] + 1
    print(x * y)