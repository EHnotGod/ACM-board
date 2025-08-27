# 对应题目：https://www.luogu.com.cn/problem/P3372
# py代码100%超时，这是洛谷的问题，不要在意，其他地方py给开额外时间的。

class Node:
    def __init__(self, l, r, he, add):
        self.l = l
        self.r = r
        self.he = he
        self.add = add
N = 100005
tr = [Node(0, 0, 0, 0) for i in range(N * 4)]
def pushdown(p):
    lc = 2 * p
    rc = 2 * p + 1
    if tr[p].add:
        tr[lc].he += tr[p].add * (tr[lc].r - tr[lc].l + 1)
        tr[rc].he += tr[p].add * (tr[rc].r - tr[rc].l + 1)
        tr[lc].add += tr[p].add
        tr[rc].add += tr[p].add
        tr[p].add = 0
def pushup(p):
    tr[p].he = tr[p * 2].he + tr[p * 2 + 1].he
def build(p, l, r):
    tr[p] = Node(l, r, w[l], 0)
    if l == r:
        return
    m = (l + r) // 2
    build(p * 2, l, m)
    build(p * 2 + 1, m + 1, r)
    pushup(p)
def update(p, x, y, k):
    if x <= tr[p].l and tr[p].r <= y:
        tr[p].he += (tr[p].r - tr[p].l + 1) * k
        tr[p].add += k
        return
    m = (tr[p].l + tr[p].r) // 2
    pushdown(p)
    if x <= m:
        update(p * 2, x, y, k)
    if y > m:
        update(p * 2 + 1, x, y, k)
    pushup(p)
def query(p, x, y):
    if x <= tr[p].l and tr[p].r <= y:
        return tr[p].he
    m = (tr[p].l + tr[p].r) // 2
    pushdown(p)
    he = 0
    if x <= m:
        he += query(p * 2, x, y)
    if y > m:
        he += query(p * 2 + 1, x, y)
    return he
n, m = map(int, input().split())
w = list(map(int, input().split()))
w.insert(0, 0)
build(1, 1, n)
for i in range(m):
    ru = list(map(int, input().split()))
    if ru[0] == 1:
        update(1, ru[1], ru[2], ru[3])
    else:
        print(query(1, ru[1], ru[2]))