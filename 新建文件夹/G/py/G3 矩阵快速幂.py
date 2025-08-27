import sys
MOD = 10**9 + 7
def mat_mult(X, Y, n):  # 矩阵乘法：返回 X * Y
    Z = [[0] * n for _ in range(n)]
    for i in range(n):
        for k in range(n):
            if X[i][k]:  # 跳过零元，加快运算
                xik = X[i][k]
                for j in range(n):
                    Z[i][j] = (Z[i][j] + xik * Y[k][j]) % MOD
    return Z

def mat_pow(A, exp, n):  # 快速幂：A^exp
    # 初始化为单位矩阵
    res = [[1 if i == j else 0 for j in range(n)] for i in range(n)]
    base = A
    while exp:
        if exp & 1:
            res = mat_mult(res, base, n)
        base = mat_mult(base, base, n)
        exp >>= 1
    return res
n, k = map(int, input().split())
# 读取矩阵，转换为 0-index 列表
A = []
for i in range(n):
    A.append(list(map(int, input().split())))
# 计算 A^k 模 MOD
result = mat_pow(A, k, n)
# 输出结果矩阵
out = []
for row in result:
    out.append(' '.join(str(v) for v in row))
sys.stdout.write('\n'.join(out))
