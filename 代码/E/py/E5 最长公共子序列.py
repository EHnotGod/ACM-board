a = "ADABBC"
b = "DBPCA"
f = []
p = []
m = 0
n = 0

def LCS():
    global m, n, f, p
    m = len(a)
    n = len(b)
    f = [[0] * (n + 1) for _ in range(m + 1)]
    p = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if a[i-1] == b[j-1]:
                f[i][j] = f[i-1][j-1] + 1
                p[i][j] = 1  # 左上方
            else:
                if f[i][j-1] > f[i-1][j]:
                    f[i][j] = f[i][j-1]
                    p[i][j] = 2  # 左边
                else:
                    f[i][j] = f[i-1][j]
                    p[i][j] = 3  # 上边
    print(f[m][n])

def getLCS():
    global m, n, p, a
    i = m
    j = n
    k = f[m][n]
    s = [''] * k
    while i > 0 and j > 0:
        if p[i][j] == 1:
            s[k-1] = a[i-1]
            i -= 1
            j -= 1
            k -= 1
        elif p[i][j] == 2:
            j -= 1
        else:
            i -= 1
    print(''.join(s))

if __name__ == "__main__":
    LCS()
    getLCS()