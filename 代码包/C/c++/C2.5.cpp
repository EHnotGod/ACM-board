// 结构体版
#include <cstring>
#include <iostream>
#include <algorithm>
using namespace std;

#define N 100005
#define LL long long
#define int long long
#define lc u<<1
#define rc u<<1|1
LL w[N];
LL n,m,op,x,y,k,mod;
struct Tree{ //线段树
  LL l,r,sum,add,mul;
}tr[N*4];

inline LL len(int u) { return tr[u].r - tr[u].l + 1; }

void pushup(int u) {
  tr[u].sum = (tr[lc].sum + tr[rc].sum) % mod;
}

void apply_mul_add_to_node(int u, LL mulv, LL addv) {
  mulv %= mod; if (mulv < 0) mulv += mod;
  addv %= mod; if (addv < 0) addv += mod;
  tr[u].sum = ( (tr[u].sum * mulv) % mod + (addv * (len(u) % mod)) % mod ) % mod;
  tr[u].mul = (tr[u].mul * mulv) % mod;
  tr[u].add = (tr[u].add * mulv + addv) % mod;
}

void pushdown(int u) {
  if (tr[u].mul != 1 || tr[u].add != 0) {
    apply_mul_add_to_node(lc, tr[u].mul, tr[u].add);
    apply_mul_add_to_node(rc, tr[u].mul, tr[u].add);
    tr[u].mul = 1;
    tr[u].add = 0;
  }
}
void build(LL u,LL l,LL r){ //建树
  tr[u]={l,r,w[l],0,1};
  if(l==r) return;
  LL m=l+r>>1;
  build(lc,l,m);
  build(rc,m+1,r);
  pushup(u);
}
void change(LL u,LL l,LL r,LL k){ //区修
  if(l<=tr[u].l&&tr[u].r<=r){
    apply_mul_add_to_node(u, 1, k);
    return;
  }
  LL m=tr[u].l+tr[u].r>>1;
  pushdown(u);
  if(l<=m) change(lc,l,r,k);
  if(r>m) change(rc,l,r,k);
  pushup(u);
}
void change2(LL u,LL l,LL r,LL k){ //区修
  if(l<=tr[u].l&&tr[u].r<=r){
    apply_mul_add_to_node(u, k, 0);
    return;
  }
  LL m=tr[u].l+tr[u].r>>1;
  pushdown(u);
  if(l<=m) change2(lc,l,r,k);
  if(r>m) change2(rc,l,r,k);
  pushup(u);
}
LL query(LL u,LL l,LL r){ //区查
  if(l<=tr[u].l && tr[u].r<=r) return tr[u].sum % mod;
  LL m=tr[u].l+tr[u].r>>1;
  pushdown(u);
  LL sum=0;
  if(l<=m) {
    sum+=query(lc,l,r);
    sum %= mod;
  }
  if(r>m) {
    sum+=query(rc,l,r);
    sum %= mod;
  }
  return sum;
}
signed main(){
  cin>>n>>m>>mod;
  for(int i=1; i<=n; i ++) cin>>w[i];
  build(1,1,n);
  while(m--){
    cin>>op>>x>>y;
    if(op==3)cout<<query(1,x,y)<<endl;
    else if (op == 2){
      cin>>k;change(1,x,y,k);
    }
    else{
      cin>>k;change2(1,x,y,k);
    }
  }
  return 0;
}