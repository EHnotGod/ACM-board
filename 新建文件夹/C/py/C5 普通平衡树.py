import sys


class Node:
    def __init__(self):
        self.ch = [0, 0]  # 左右孩子
        self.fa = 0  # 父节点
        self.v = 0  # 节点值
        self.cnt = 0  # 值出现次数
        self.siz = 0  # 子树大小

    def init(self, p, v1):
        self.fa = p
        self.v = v1
        self.cnt = self.siz = 1


def ls(x):
    return tr[x].ch[0]


def rs(x):
    return tr[x].ch[1]


def pushup(x):
    tr[x].siz = tr[ls(x)].siz + tr[rs(x)].siz + tr[x].cnt


def rotate(x):
    y = tr[x].fa
    z = tr[y].fa
    k = 1 if tr[y].ch[1] == x else 0
    tr[z].ch[1 if tr[z].ch[1] == y else 0] = x
    tr[x].fa = z
    tr[y].ch[k] = tr[x].ch[k ^ 1]
    tr[tr[x].ch[k ^ 1]].fa = y
    tr[x].ch[k ^ 1] = y
    tr[y].fa = x
    pushup(y)
    pushup(x)


def splay(x, k):
    while tr[x].fa != k:
        y = tr[x].fa
        z = tr[y].fa
        if z != k:
            if (ls(y) == x) ^ (ls(z) == y):
                rotate(x)
            else:
                rotate(y)
        rotate(x)
    if k == 0:
        global root
        root = x


def insert(v):
    global root, tot
    x = root
    p = 0
    while x != 0 and tr[x].v != v:
        p = x
        x = tr[x].ch[1 if v > tr[x].v else 0]
    if x != 0:
        tr[x].cnt += 1
    else:
        tot += 1
        x = tot
        tr[p].ch[1 if v > tr[p].v else 0] = x
        tr[x].init(p, v)
    splay(x, 0)


def find(v):
    global root
    x = root
    while tr[x].ch[1 if v > tr[x].v else 0] != 0 and v != tr[x].v:
        x = tr[x].ch[1 if v > tr[x].v else 0]
    splay(x, 0)


def getpre(v):
    global root
    find(v)
    x = root
    if tr[x].v < v:
        return x
    x = ls(x)
    while rs(x) != 0:
        x = rs(x)
    splay(x, 0)
    return x


def getsuc(v):
    global root
    find(v)
    x = root
    if tr[x].v > v:
        return x
    x = rs(x)
    while ls(x) != 0:
        x = ls(x)
    splay(x, 0)
    return x


def del_(v):
    pre = getpre(v)
    suc = getsuc(v)
    splay(pre, 0)
    splay(suc, pre)
    del_node = tr[suc].ch[0]
    if tr[del_node].cnt > 1:
        tr[del_node].cnt -= 1
        splay(del_node, 0)
    else:
        tr[suc].ch[0] = 0
        splay(suc, 0)


def getrank(v):
    insert(v)
    res = tr[ls(root)].siz
    del_(v)
    return res


def getval(k):
    global root
    x = root
    while True:
        if k <= tr[ls(x)].siz:
            x = ls(x)
        elif k <= tr[ls(x)].siz + tr[x].cnt:
            break
        else:
            k -= tr[ls(x)].siz + tr[x].cnt
            x = rs(x)
    splay(x, 0)
    return tr[x].v


N = 110001
INF = (1 << 30) + 1
tr = [Node() for _ in range(N)]
root = 0
tot = 0


insert(-INF)
insert(INF)
n = int(sys.stdin.readline())
for _ in range(n):
    op, x = map(int, sys.stdin.readline().split())
    if op == 1:
        insert(x)
    elif op == 2:
        del_(x)
    elif op == 3:
        print(getrank(x))
    elif op == 4:
        print(getval(x + 1))
    elif op == 5:
        print(tr[getpre(x)].v)
    else:
        print(tr[getsuc(x)].v)
