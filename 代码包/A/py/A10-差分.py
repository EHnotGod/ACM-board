# P4552 [Poetize6] IncDec Sequence

n = int(input())
a = [0]
b = [0] * (n + 1)
for i in range(1, n + 1):
    a.append(int(input()))
    b[i] = a[i] - a[i - 1]
p, q = 0, 0
for i in range(2, n + 1):
    if b[i] > 0:
        p += b[i]
    else:
        q += abs(b[i])
print(max(p, q))
print(abs(p - q) + 1)