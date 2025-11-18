#include <bits/stdc++.h>
using namespace std;
const int N = 2010;
int n, m;
int d[N], vis[N], cnt[N];
struct Edge {
    int v, w;
};
vector<Edge> g[N];
bool spfa() { // 判负环
    memset(d, 0x3f, sizeof d); d[1] = 0;
    memset(vis, 0, sizeof vis);
    memset(cnt, 0, sizeof cnt);
    queue<int> q;
    q.push(1);
    vis[1] = 1; // 在队中
    while (!q.empty()) {
        int u = q.front(); q.pop();
        vis[u] = 0;
        for (auto &e : g[u]) {
            int v = e.v, w = e.w;
            if (d[v] > d[u] + w) { // 松弛
                d[v] = d[u] + w;
                cnt[v] = cnt[u] + 1; // 边数
                if (cnt[v] >= n) return true; // 有负环
                if (!vis[v]) {
                    q.push(v);
                    vis[v] = 1;
                }
            }
        }
    }
    return false;
}

int main() {
    int T;
    scanf("%d", &T);
    while (T--) {
        scanf("%d%d", &n, &m);
        for (int i = 1; i <= n; i++) g[i].clear();
        for (int i = 1; i <= m; i++) {
            int u, v, w;
            scanf("%d%d%d", &u, &v, &w);
            g[u].push_back({v, w});
            if (w >= 0) g[v].push_back({u, w});
        }
        puts(spfa() ? "YES" : "NO");
    }
}