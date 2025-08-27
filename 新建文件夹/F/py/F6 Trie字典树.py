N = 100010
ch = [[0 for _ in range(26)] for _ in range(N)]
cnt = [0] * N
idx = 0

def insert(s):
    global idx
    p = 0
    for c in s:
        j = ord(c) - ord('a')
        if not ch[p][j]:
            idx += 1
            ch[p][j] = idx
        p = ch[p][j]
    cnt[p] += 1

def query(s):
    p = 0
    for c in s:
        j = ord(c) - ord('a')
        if not ch[p][j]:
            return 0
        p = ch[p][j]
    return cnt[p]

n, q = map(int, input().split())
for _ in range(n):
    s = input().strip()
    insert(s)
for _ in range(q):
    s = input().strip()
    print(query(s))