#include <bits/stdc++.h>
using namespace std;
typedef long long ll;
typedef unsigned long long ull;

int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);
    int t;
    cin >> t;

    mt19937_64 rng(random_device{}()); // 64位随机生成器

    while (t--) {
        int n;
        cin >> n;
        vector<ll> l(n), r(n);
        vector<ull> w(n);
        for (int i = 0; i < n; i++) {
            w[i] = rng(); // 生成随机权值
        }

        map<ll, ull> event_map; // 记录事件: {坐标, 异或值}

        for (int i = 0; i < n; i++) {
            cin >> l[i] >> r[i];
            event_map[l[i]] ^= w[i];          // 在l[i]加入
            event_map[r[i] + 1] ^= w[i];      // 在r[i]+1移除
        }

        set<ull> seen = {0}; // 覆盖状态集合，初始化空状态
        ull cur = 0;         // 当前覆盖状态
        vector<ll> poses;    // 事件坐标
        for (auto& p : event_map) {
            poses.push_back(p.first);
        }
        sort(poses.begin(), poses.end()); // 坐标排序

        for (ll pos : poses) {
            cur ^= event_map[pos];      // 更新当前状态
            seen.insert(cur);            // 记录新状态
        }

        cout << seen.size() << '\n';
    }
    return 0;
}