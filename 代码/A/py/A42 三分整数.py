def get_sum(f, x):
    total = 0
    for k, a, b in f:
        total += abs(k * x + a) + b
    return total

t = int(input())
for _ in range(t):
    n, l, r = map(int, input().split())
    f = [tuple(map(int, input().split())) for _ in range(n)]

    while l + 2 <= r:
        diff = (r - l) // 3
        mid1 = l + diff
        mid2 = r - diff
        sum1 = get_sum(f, mid1)
        sum2 = get_sum(f, mid2)
        if sum1 <= sum2:
            r = mid2 - 1
        else:
            l = mid1 + 1

    print(min(get_sum(f, l), get_sum(f, r)))
