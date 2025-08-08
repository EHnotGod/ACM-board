n, q = map(int, input().split())

v = [0] * (n + 1)
primes = []

for i in range(2, n + 1):
    if v[i] == 0:  # i 是素数
        v[i] = i
        primes.append(i)
    for p in primes:
        m = i * p
        if m > n:
            break
        v[m] = p  # 记录最小质因数
        if i % p == 0:
            break
for _ in range(q):
    k = int(input())
    print(v[k] if k > 1 else 1)
