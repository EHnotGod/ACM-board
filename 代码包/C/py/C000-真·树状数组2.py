import sys
input = sys.stdin.readline

class BIT:
    def __init__(self, n):
        self.n = n
        self.s = [0] * (n + 1)

    def change(self, x, w):
        while x <= self.n:
            self.s[x] += w
            x += x & -x

    def query(self, x):
        res = 0
        while x > 0:
            res += self.s[x]
            x -= x & -x
        return res

if __name__ == "__main__":
    n, m = map(int, input().split())
    T = BIT(n)
    a = list(map(int, input().split()))
    prev = 0
    for i, val in enumerate(a, 1):  # 下标从1开始
        d = val - prev
        T.change(i, d)
        prev = val
    for _ in range(m):
        op, *rest = map(int, input().split())
        if op == 1:  # 区间修改： op=1 l r k
            l, r, k = rest
            T.change(l, k)
            if r + 1 <= n:
                T.change(r + 1, -k)
        else:        # 点查询： op=2 x
            x = rest[0]
            print(T.query(x))