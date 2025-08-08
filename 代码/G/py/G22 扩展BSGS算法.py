import math

def gcd(a, b):
    return a if b == 0 else gcd(b, a % b)

def exbsgs(a, b, p):
    a %= p
    b %= p
    if b == 1 or p == 1:
        return 0  # x = 0

    # 处理 gcd 除掉公共因子
    k = 0
    A = 1 % p
    while True:
        d = gcd(a, p)
        if d == 1:
            break
        if b % d != 0:
            return -1
        k += 1
        b //= d
        p //= d
        A = (A * (a // d)) % p
        if A == b:
            return k

    # baby-step giant-step
    m = math.isqrt(p)
    if m * m < p:
        m += 1

    t = b % p
    table = {t: 0}
    for j in range(1, m):
        t = (t * a) % p
        table[t] = j

    mi = pow(a, m, p)  # a^m % p
    t = A % p
    for i in range(1, m + 1):
        t = (t * mi) % p
        if t in table:
            return i * m - table[t] + k

    return -1

# 使用 input() 逐个 token 读取（可处理不同换行/空格的输入格式）
def tokens():
    while True:
        try:
            for tok in input().split():
                yield tok
        except EOFError:
            return

if __name__ == "__main__":
    it = tokens()
    while True:
        try:
            a = int(next(it))
            p = int(next(it))
            b = int(next(it))
        except StopIteration:
            break
        if a == 0:
            break
        res = exbsgs(a, b, p)
        if res == -1:
            print("No Solution")
        else:
            print(res)
