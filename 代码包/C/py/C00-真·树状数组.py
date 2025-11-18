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
    for i, k in enumerate(a, 1):  # 下标从1开始
        T.change(i, k)

    for _ in range(m):
        op, *rest = map(int, input().split())
        if op == 1:  # 单点修改
            x, k = rest
            T.change(x, k)
        else:        # 区间查询
            x, y = rest
            print(T.query(y) - T.query(x - 1))
