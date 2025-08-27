n = int(input())
s = []
for i in range(n):
    s.append(list(map(int, input().split())))

dp = [[1145145 ** 5 for i in range(n)] for _ in range(2 ** n)]
dp[2 ** 0][0] = 0
for i in range(2 ** n):
    for j in range(n):
        if i & 2 ** j:
            for k in range(n):
                if i & 2 ** k and k != j:
                    dp[i][j] = min(dp[i][j], dp[i ^ 2 ** j][k] + s[k][j])
# 为了回到原点
ans = min(dp[2 ** n - 1][j] + s[j][0] for j in range(1, n))
print(ans)