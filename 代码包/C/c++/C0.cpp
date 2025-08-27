/*EHnotgod————
..............#######.....#.....#..............
..............#...........#.....#..............
..............#######.....#######..............
..............#...........#.....#..............
..............#######.....#.....#..............
*/

#include<bits/stdc++.h>
using namespace std;
#define endl "\n"

#define range(i, a, b) for (int i = (a); i < (b); ++i)

#define lc (p<<1)
#define rc (p<<1|1)
#define N 500005

int n, w[N];

struct node {
    int l, r, sum;
} tr[N * 4];

void build(int p, int l, int r) {
    tr[p].l = l;
    tr[p].r = r;
    tr[p].sum = w[l];
    if (l == r) return;
    int m = (l + r) / 2;
    build(lc, l, m);
    build(rc, m + 1, r);
    tr[p].sum = tr[lc].sum + tr[rc].sum;
}

// 点修改（从根递归进入）
void update(int p, int x, int k) {
    if (tr[p].l == x && tr[p].r == x) {
        tr[p].sum += k;
        return;
    }
    int m = (tr[p].l + tr[p].r) / 2;
    if (x <= m) update(lc, x, k);
    else update(rc, x, k);
    tr[p].sum = tr[lc].sum + tr[rc].sum;
}

int query(int p, int x, int y) {
    if (x <= tr[p].l && tr[p].r <= y) return tr[p].sum;
    int m = (tr[p].l + tr[p].r) / 2;
    int sum = 0;
    if (x <= m) sum += query(lc, x, y);
    if (y > m)  sum += query(rc, x, y);
    return sum;
}
int main(){
    ios::sync_with_stdio(0);
    cin.tie(0);
    cout.tie(0);
    int n, m;
    cin >> n >> m;
    range(i, 0, n){
        cin >> w[i + 1];
    }
    build(1, 1, n);
    range(i, 0, m){
        int op; cin >> op;
        int x, y, k;
        if (op == 1){
            cin >> x >> k;  // 单点更新
            update(1, x, k);
        }
        else{
            cin >> x >> y;  // 区间查询
            cout << query(1, x, y) << endl;
        }
    }
}