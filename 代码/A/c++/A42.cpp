#include <iostream>
#include <vector>
using namespace std;

struct fun {
    int k, a, b;
};

long long getSum(vector<fun>& f, long long x) {
    long long ans = 0;
    for (fun& fi : f) {
        ans += abs(fi.k * x + fi.a) + fi.b;
    }
    return ans;
}

int main() {
    int t, n, l, r;
    cin >> t;
    while (t--) {
        cin >> n >> l >> r;
        vector<fun> f(n);
        for (int i = 0; i < n; ++i) {
            cin >> f[i].k >> f[i].a >> f[i].b;
        }
        while (l + 2 <= r) {
            int diff = (r - l) / 3;
            int mid1 = l + diff;
            int mid2 = r - diff;
            long long sum1 = getSum(f, mid1);
            long long sum2 = getSum(f, mid2);
            if (sum1 <= sum2) {
                r = mid2 - 1;
            } else {
                l = mid1 + 1;
            }
        }
        cout << min(getSum(f, l), getSum(f, r)) << endl;
    }
}
// 64 位输出请用 printf("%lld")