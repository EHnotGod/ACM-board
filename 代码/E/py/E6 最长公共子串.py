a = "BCCABCCB"
b = "AACCAB"
m = len(a)
n = len(b)
f = [[0] * (n + 1) for _ in range(m + 1)]
ans = 0
for i in range(1, m + 1):
    for j in range(1, n + 1):
        if a[i - 1] == b[j - 1]:
            f[i][j] = f[i - 1][j - 1] + 1
        else:
            f[i][j] = 0
        ans = max(ans, f[i][j])
print(ans)