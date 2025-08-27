#include <bits/stdc++.h>
using namespace std;
int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);
    int n;
    cin >> n;
    vector<vector<int>> s(n, vector<int>(n));
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            cin >> s[i][j];
        }
    }
    const int INF = 1e9;  // 比较大的数即可
    vector<vector<int>> dp(1 << n, vector<int>(n, INF));
    dp[1][0] = 0; // 只访问了点0，最后停在0
    for (int mask = 0; mask < (1 << n); mask++) {
        for (int j = 0; j < n; j++) {
            if (mask & (1 << j)) { // j在集合中
                for (int k = 0; k < n; k++) {
                    if ((mask & (1 << k)) && k != j) {
                        dp[mask][j] = min(dp[mask][j],
                                          dp[mask ^ (1 << j)][k] + s[k][j]);
                    }
                }
            }
        }
    }
    int ans = INF;
    for (int j = 1; j < n; j++) {
        ans = min(ans, dp[(1 << n) - 1][j] + s[j][0]);
    }
    cout << ans << "\n";
    return 0;
}
