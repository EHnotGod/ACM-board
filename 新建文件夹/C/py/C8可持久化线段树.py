import bisect

class Node:
    def __init__(self, l=0, r=0, s=0):
        self.l = l
        self.r = r
        self.s = s


n, m = map(int, input().split())
a = list(map(int, input().split()))
a = [0] + a  # 转换为1-based索引

# 离散化处理
sorted_b = sorted(a[1:])
unique_b = []
prev = None
for num in sorted_b:
    if num != prev:
        unique_b.append(num)
        prev = num
bn = len(unique_b)

# 初始化主席树
tr = [Node(0, 0, 0)]  # 空节点，索引0
idx = 1
root = [0] * (n + 1)
root[0] = 0

def insert(x, l, r, pos):
    global idx
    y = Node(tr[x].l, tr[x].r, tr[x].s + 1)
    tr.append(y)
    y_idx = idx
    idx += 1
    if l == r:
        return y_idx
    mid = (l + r) // 2
    if pos <= mid:
        new_l = insert(tr[x].l, l, mid, pos)
        y.l = new_l
    else:
        new_r = insert(tr[x].r, mid + 1, r, pos)
        y.r = new_r
    return y_idx

# 构建主席树的每个版本
for i in range(1, n + 1):
    num = a[i]
    pos = bisect.bisect_left(unique_b, num) + 1  # 转换为1-based的id
    root[i] = insert(root[i-1], 1, bn, pos)

# 查询函数
def query(x, y, l, r, k):
    if l == r:
        return l
    mid = (l + r) // 2
    left_x = tr[x].l
    left_y = tr[y].l
    s = tr[left_y].s - tr[left_x].s
    if k <= s:
        return query(left_x, left_y, l, mid, k)
    else:
        return query(tr[x].r, tr[y].r, mid + 1, r, k - s)

# 处理每个查询并输出结果
output = []
for _ in range(m):
    l, r, k = map(int, input().split())
    id = query(root[l-1], root[r], 1, bn, k)
    output.append(str(unique_b[id-1]))  # id转换为0-based索引
print('\n'.join(output))