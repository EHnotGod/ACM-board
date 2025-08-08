import sys

input = sys.stdin.readline
sys.setrecursionlimit(10 ** 6)


class Node():
    def __init__(self, l, r, he):
        self.l = l
        self.r = r
        self.he = he


def build(p, l, r):
    tr[p] = Node(l, r, 0)
    if l == r:
        tr[p].he = w[l]
        return

    m = (l + r) // 2

    build(p * 2, l, m)
    build(p * 2 + 1, m + 1, r)
    tr[p].he = tr[p * 2].he + tr[p * 2 + 1].he


def update(p, idx, k):
    if tr[p].l == idx and tr[p].r == idx:
        tr[p].he += k
        return
    m = (tr[p].l + tr[p].r) // 2

    if idx <= m:
        update(p * 2, idx, k)
    if idx > m:
        update(p * 2 + 1, idx, k)
    tr[p].he = tr[p * 2].he + tr[p * 2 + 1].he


def query(p, l, r):
    if tr[p].l >= l and tr[p].r <= r:
        return tr[p].he
    m = (tr[p].l + tr[p].r) // 2
    he = 0
    if l <= m:
        he += query(p * 2, l, r)
    if r > m:
        he += query(p * 2 + 1, l, r)
    return he


n, m = map(int, input().split())
w = list(map(int, input().split()))
N = n * 4 + 1
tr = [Node(0, 0, 0) for _ in range(N * 4)]
build(1, 0, n - 1)
for i in range(m):
    tmp = list(map(int, input().split()))
    if tmp[0] == 2:
        print(query(1, tmp[1] - 1, tmp[2] - 1))
    else:
        update(1, tmp[1] - 1, tmp[2])
