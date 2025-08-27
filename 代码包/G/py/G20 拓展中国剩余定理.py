def exgcd(a, b):
    if b == 0:
        return a, 1, 0
    d, x1, y1 = exgcd(b, a % b)
    x = y1
    y = x1 - a // b * y1
    return d, x, y

def EXCRT(m, r):
    m1 = m[1]
    r1 = r[1]
    for i in range(2, n + 1):
        m2 = m[i]
        r2 = r[i]
        d, p, q = exgcd(m1, m2)
        # 不可整除则无解
        if (r2 - r1) % d != 0:
            return -1
        # 求一个特解并取模
        p = p * ((r2 - r1) // d)
        p = p % (m2 // d)
        r1 = m1 * p + r1
        m1 = m1 * (m2 // d)
    return r1 % m1

if __name__ == "__main__":
    n = int(input().strip())
    m = [0] * (n + 1)
    r = [0] * (n + 1)
    for i in range(1, n + 1):
        mi, ri = map(int, input().split())
        m[i] = mi
        r[i] = ri
    print(EXCRT(m, r))
