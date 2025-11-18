#include <bits/stdc++.h>
using namespace std;
#define endl "\n"
#define int long long
#define range(i, a, b) for (int i = (a); i < (b); ++i)
#define def(name, ...) auto name = [&](__VA_ARGS__)
vector<vector<pair<int, int>>> e;
// vector<int> a(n, 0);
// vector<vector<int>> dp(n + 1, vector<int>(n + 1, 0));
void solve() {
	int n, m, k; cin >> n >> m >> k;
    e.assign(n * m, vector<pair<int, int>>());
    range(i, 0, n){
        string s; cin >> s;
        range(j, 0, m){
            int idx = i * m + j;
            if (i > 0){
                if (s[j] == 'U')e[idx].push_back({idx - m, 0});
                else e[idx].push_back({idx - m, 1});
            }
            if (j > 0){
                if (s[j] == 'L')e[idx].push_back({idx - 1, 0});
                else e[idx].push_back({idx - 1, 1});
            }
            if (i < n - 1){
                if (s[j] == 'D')e[idx].push_back({idx + m, 0});
                else e[idx].push_back({idx + m, 1});
            }
            if (j < m - 1){
                if (s[j] == 'R')e[idx].push_back({idx + 1, 0});
                else e[idx].push_back({idx + 1, 1});
            }
        }
    }
    deque<pair<int, int>> q;
    vector<int> vis(n * m, -1);
    q.push_back({0, 0});
    while (q.size()){
        auto [u, dis] = q.front();
        q.pop_front();
        if (vis[u] != -1) continue;
        vis[u] = dis;
        for (auto[v, le]: e[u]){
            if (vis[v] == -1){
                if (le == 0){
                    q.push_front({v, dis});
                }
                else {
                    q.push_back({v, dis + 1});
                }
            }
        }
    }
    if (vis[n*m-1] > k){
        cout << "NO" << endl;
    }
    else {
        cout << "YES" << endl;
    }
}

signed main() {
    ios::sync_with_stdio(0);
    cin.tie(0);
    cout.tie(0);
	int t;
	cin >> t;
	while (t--) {
		solve();
	}
	return 0;
}