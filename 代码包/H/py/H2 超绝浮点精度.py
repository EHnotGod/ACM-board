from decimal import Decimal, getcontext

getcontext().prec = 200

n, k = map(int, input().split())
n = Decimal(n)
k = Decimal(k)

ans1 = Decimal(0)
for i in range(int(k)):
    ans1 += k / Decimal(i + 1)

ans2 = k * (Decimal(1) - (Decimal(1) - Decimal(1) / k) ** n)

print(ans1, ans2)