# 对应题目：https://www.luogu.com.cn/problem/P1496
# py代码大概率超时，这是洛谷的问题，不要在意，其他地方py给开额外时间的。

n = int(input())
lis = []
for i in range(n):
    a, b = map(int, input().split())
    lis.append([a, b])
lis.sort(key=lambda x: (x[0], x[1]))
l = -1e18
r = -1e18
ans = 0
for i in range(n):
    a, b = lis[i]
    if r < a:
        ans += r - l
        l = a
        r = b
    else:
        r = max(r, b)
ans += r - l
print(int(ans))