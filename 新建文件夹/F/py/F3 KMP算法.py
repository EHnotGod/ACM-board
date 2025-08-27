N = 1000010
S = list(input().strip())
S = [""] + S
P = list(input().strip())
P = [""] + P
nxt = [0] * N
m = len(S) - 1
n = len(P) - 1
S.append("")
P.append("")
nxt[1] = 0
j = 0
for i in range(2, n + 1):
    while j and P[i] != P[j + 1]:
        j = nxt[j]
    if P[i] == P[j + 1]:
        j += 1
    nxt[i] = j
j = 0
for i in range(1, m + 1):
    while j and S[i] != P[j + 1]:
        j = nxt[j]
    if S[i] == P[j + 1]:
        j += 1
    if j == n:
        print(i - n + 1)
print(*nxt[1:n + 1])