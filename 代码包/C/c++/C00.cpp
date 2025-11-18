#include <bits/stdc++.h>
using namespace std;

const int N = 500010;
int n, m;

struct BIT {
    vector<int> s; // 用 vector 更灵活
    BIT(int n = 0) { s.assign(n + 1, 0); }
    void change(int x, int w) {
        for (; x < (int)s.size(); x += x & -x) s[x] += w;
    }
    int query(int x) {
        int res = 0;
        for (; x; x -= x & -x) res += s[x];
        return res;
    }
};

int main() {
    ios::sync_with_stdio(0);
    cin.tie(0);

    cin >> n >> m;
    BIT T(n);  // 局部创建，大小为 n

    int op, x, y, k;
    for (int i = 1; i <= n; i++) {
        cin >> k;
        T.change(i, k);
    }

    for (int i = 1; i <= m; i++) {
        cin >> op >> x;
        if (op == 1) {
            cin >> k;
            T.change(x, k);
        } else {
            cin >> y;
            cout << T.query(y) - T.query(x - 1) << "\n";
        }
    }
}
