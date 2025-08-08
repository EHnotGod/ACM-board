import sys
N = 500
def mul(a, b):  # 高精度乘法
    t = [0] * (N * 2)
    for i in range(N):
        for j in range(N):
            t[i + j] += a[i] * b[j]
            t[i + j + 1] += t[i + j] // 10
            t[i + j] %= 10
    return t
def quickpow(p):  # 快速幂
    a = [0] * N
    res = [0] * N
    a[0] = 2
    res[0] = 1
    while p:
        if p & 1:
            res = mul(res, a)
        a = mul(a, a)
        p >>= 1
    res[0] -= 1  # 个位修正
    return res
def main():
    p = int(sys.stdin.readline())
    # 输出位数
    print(int(p * __import__('math').log10(2)) + 1)

    res = quickpow(p)
    for i in range(10):
        start = N - 1 - i * 50
        line_digits = res[start - 49:start + 1]
        print(''.join(str(d) for d in reversed(line_digits)))
if __name__ == '__main__':
    main()
