import sys

def get_phi(n):
    phi = [0] * (n + 1)
    vis = [False] * (n + 1)
    primes = []
    phi[1] = 1
    for i in range(2, n + 1):
        if not vis[i]:
            primes.append(i)
            phi[i] = i - 1
        for p in primes:
            m = i * p
            if m > n:
                break
            vis[m] = True
            if i % p == 0:
                # p | i
                phi[m] = p * phi[i]
                break
            else:
                phi[m] = (p - 1) * phi[i]
    return phi

data = sys.stdin.read().split()
n = int(data[0])
phi = get_phi(n)
# 输出 1..n
out = '\n'.join(str(phi[i]) for i in range(1, n + 1))
sys.stdout.write(out)
