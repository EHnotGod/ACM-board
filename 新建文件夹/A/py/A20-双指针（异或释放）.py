# https://www.luogu.com.cn/problem/AT_arc098_b

n = int(input())
a = list(map(int, input().split()))
a.insert(0, 0)
s1 = 0
s2 = 0
ans = 0
i = 1
j = 0
while i <= n:
    while j + 1 <= n and s1 + a[j + 1] == (s2 ^ a[j + 1]):
        j += 1
        s1 += a[j]
        s2 ^= a[j]
    ans += j - i + 1
    s1 -= a[i]
    s2 -= a[i]
    i += 1
print(ans)