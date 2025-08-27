#include "bits/stdc++.h"
using namespace std;
using ll = long long;
using lll = __int128;
#define all(x) (x).begin(),(x).end()
const ll p = 998244353;
ll ksm(ll x, lll Y)
{
	ll r = 1;
	x %= p;
	if (!Y) return 1;
	if (!x) return 0;
	ll y = Y % (p - 1);
	while (y)
	{
		if (y & 1) r = r * x % p;
		x = x * x % p; y >>= 1;
	}
	return r;
}
ll f(ll a, lll b, ll c, lll d)
{
	ll x = gcd(a, c);
	if (x == 1 || b == 0 || d == 0) return 1;
	lll ka = 0, kc = 0;
	while (a % x == 0) a /= x, ++ka;
	while (c % x == 0) c /= x, ++kc;
	ka *= b, kc *= d;
	if (ka > kc) swap(a, c), swap(b, d), swap(ka, kc);
	return ksm(x, ka) * f(a, b, x, kc - ka) % p;
}
int main()
{
	ios::sync_with_stdio(0);
	cin.tie(0);
	int T;
	cin >> T;
	while (T--)
	{
		ll a, b, c, d;
		cin >> a >> b >> c >> d;
		cout << f(a, b, c, d) << '\n';
	}
}