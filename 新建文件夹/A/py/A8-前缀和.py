# P8218 【深进1.例1】求区间和

n = int(input())
a = list(map(int, input().split()))
s = [0] * (n + 1)
for i in range(1, n + 1):
    s[i] = s[i - 1] + a[i - 1]
m = int(input())
for i in range(m):
    l, r = map(int, input().split())
    print(s[r] - s[l - 1])