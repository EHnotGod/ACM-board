import sys
import threading
import math

MOD = 998244353
PHI = MOD - 1  # 欧拉函数 phi(998244353)

def qmi(a, k):
    res = 1
    a %= MOD
    while k:
        if k & 1:
            res = res * a % MOD
        a = a * a % MOD
        k >>= 1
    return res

def dfs(a, b, c, d):
    g = math.gcd(a, c)
    if g == 1:
        return 1
    minv = min(b, d)
    if b <= d:
        exp = b if b < PHI else b % PHI + PHI
        return qmi(g, exp) * dfs(a // g, b, g, d - b) % MOD
    else:
        exp = d if d < PHI else d % PHI + PHI
        return qmi(g, exp) * dfs(g, b - d, c // g, d) % MOD

def main():
    T = int(sys.stdin.readline())
    for _ in range(T):
        a, b, c, d = map(int, sys.stdin.readline().split())
        print(dfs(a, b, c, d))

threading.Thread(target=main).start()