// 结构体版
#include <cstring>
#include <iostream>
#include <algorithm>
using namespace std;

#define N 100005
#define LL long long
#define lc u<<1
#define rc u<<1|1
LL w[N];
struct Tree{ //线段树
  LL l,r,sum,add;
}tr[N*4];

void pushup(LL u){ //上传
  tr[u].sum=tr[lc].sum+tr[rc].sum;
}
void pushdown(LL u){ //下传
  if(tr[u].add){
    tr[lc].sum+=tr[u].add*(tr[lc].r-tr[lc].l+1),
    tr[rc].sum+=tr[u].add*(tr[rc].r-tr[rc].l+1),
    tr[lc].add+=tr[u].add,
    tr[rc].add+=tr[u].add,
    tr[u].add=0;
  }
}
void build(LL u,LL l,LL r){ //建树
  tr[u]={l,r,w[l],0};
  if(l==r) return;
  LL m=l+r>>1;
  build(lc,l,m);
  build(rc,m+1,r);
  pushup(u);
}
void change(LL u,LL l,LL r,LL k){ //区修
  if(l<=tr[u].l&&tr[u].r<=r){
    tr[u].sum+=(tr[u].r-tr[u].l+1)*k;
    tr[u].add+=k;
    return;
  }
  LL m=tr[u].l+tr[u].r>>1;
  pushdown(u);
  if(l<=m) change(lc,l,r,k);
  if(r>m) change(rc,l,r,k);
  pushup(u);
}
LL query(LL u,LL l,LL r){ //区查
  if(l<=tr[u].l && tr[u].r<=r) return tr[u].sum;
  LL m=tr[u].l+tr[u].r>>1;
  pushdown(u);
  LL sum=0;
  if(l<=m) sum+=query(lc,l,r);
  if(r>m) sum+=query(rc,l,r);
  return sum;
}
int main(){
  LL n,m,op,x,y,k;
  cin>>n>>m;
  for(int i=1; i<=n; i ++) cin>>w[i];

  build(1,1,n);
  while(m--){
    cin>>op>>x>>y;
    if(op==2)cout<<query(1,x,y)<<endl;
    else cin>>k,change(1,x,y,k);
  }
  return 0;
}