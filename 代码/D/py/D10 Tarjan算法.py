import sys
sys.setrecursionlimit(10**7)
input = sys.stdin.readline

# 最大节点数、查询数上限
N = 500_005
M = 2 * N

# 树的邻接表
edges = [[] for _ in range(N)]
# 存放每个节点的 LCA 查询：queries[u] = [(v, idx), ...]
queries = [[] for _ in range(N)]
# 并查集父指针，初始时 fa[x]=x
fa = list(range(N))
# 标记节点是否已访问过
visited = [False] * N
# 存放每个查询的答案，按输入顺序
ans = [0] * M

def find(x: int) -> int:
    """并查集查找（带路径压缩）"""
    if fa[x] != x:
        fa[x] = find(fa[x])
    return fa[x]

def tarjan(u: int) -> None:
    """
    以 u 为根做一次 DFS，离开子树时合并并查集，
    并在回溯到 u 时处理所有与 u 有关的 LCA 查询。
    """
    visited[u] = True
    for v in edges[u]:
        if not visited[v]:
            tarjan(v)
            # 子树处理完后，将子节点 v 的并查集父指向 u
            fa[v] = u

    # 所有子节点都已处理，处理 u 发起的所有查询
    for v, idx in queries[u]:
        # 如果查询的另一个节点 v 已经访问过，就能确定 LCA
        if visited[v]:
            ans[idx] = find(v)

# ------------------------------
# 读取输入、构建数据结构、执行 Tarjan 离线 LCA
# ------------------------------
n, m, s = map(int, input().split())

# 读入 n−1 条边，构建无向树
for _ in range(n - 1):
    a, b = map(int, input().split())
    edges[a].append(b)
    edges[b].append(a)

# 读入 m 条 LCA 查询，双向加入查询列表
for i in range(1, m + 1):
    u, v = map(int, input().split())
    queries[u].append((v, i))
    queries[v].append((u, i))

# 初始化并查集（只需对 1..n）
for i in range(1, n + 1):
    fa[i] = i

# 从根节点 s 开始 Tarjan 算法
tarjan(s)

# 输出所有查询的结果
for i in range(1, m + 1):
    print(ans[i])
