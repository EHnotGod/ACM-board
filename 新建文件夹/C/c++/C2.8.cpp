/*EHnotgod————
..............#######.....#.....#..............
..............#...........#.....#..............
..............#######.....#######..............
..............#...........#.....#..............
..............#######.....#.....#..............
*/

#include<bits/stdc++.h>
#define int long long
using namespace std;
#define endl "\n"
#define range(i, a, b) for (int i = (a); i < (b); ++i)

#define lc 2*p
#define rc 2*p+1
#define N 500005

int w[N];

struct node {
    int l, r, sum, ansl, ansr, ans;
} tr[N * 4];
void merge(int p){
    tr[p].sum = tr[lc].sum + tr[rc].sum;
    tr[p].ans = max(tr[lc].ans, max(tr[rc].ans, tr[rc].ansl + tr[lc].ansr));
    tr[p].ansl = max(tr[lc].sum + tr[rc].ansl, tr[lc].ansl);
    tr[p].ansr = max(tr[rc].sum + tr[lc].ansr, tr[rc].ansr);
}
void build(int p, int l, int r) {
    tr[p].l = l; tr[p].r = r;
    if (l == r) {
        tr[p].sum = w[l];
        tr[p].ansl = tr[p].ansr = tr[p].ans = w[l];
        return;
    }
    int m = (l + r) / 2;
    build(lc, l, m);
    build(rc, m + 1, r);
    merge(p);
}
void update(int p, int x, int k) {
    if (tr[p].l == x && tr[p].r == x) {
        tr[p].sum = k;
        tr[p].ansl = k;
        tr[p].ansr = k;
        tr[p].ans = k;
        return;
    }
    int m = (tr[p].l + tr[p].r) / 2;
    if (x <= m) update(lc, x, k);
    else update(rc, x, k);
    merge(p);
}

node query(int p, int x, int y) {
    if (x <= tr[p].l && tr[p].r <= y) return tr[p];
    int m = (tr[p].l + tr[p].r) / 2;
    if (y <= m){
        return query(lc, x, y);
    }
    else if (x > m) {
        return query(rc, x, y);
    }
    else{
        node t, a = query(lc, x, y), b = query(rc, x, y);
        t.sum = a.sum + b.sum;
        t.ansl = max(a.ansl, a.sum + b.ansl);
        t.ansr = max(b.ansr, b.sum + a.ansr);
        t.ans = max({a.ans, b.ans, a.ansr + b.ansl});
        return t;
    }
}
signed main(){
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
        if (op == 2){
            cin >> x >> k;
            update(1, x, k);
        }
        else{
            cin >> x >> y;
            if (x > y){
                cout << query(1, y, x).ans << endl;
            }
            else{
                cout << query(1, x, y).ans << endl;
            }
        }
    }
}