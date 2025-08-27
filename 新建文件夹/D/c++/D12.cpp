// 树链剖分 O(mlognlogn)
#include<bits/stdc++.h>
using namespace std;

#define int long long
const int N=100010;
int n,m,root,P,w[N],a,b,c,t;
vector<int> e[N];
int fa[N],dep[N],siz[N],son[N];
int top[N],id[N],nw[N],cnt; //树链

#define lc u<<1
#define rc u<<1|1
struct tree{
  int l,r;
  int sum,add;
}tr[N*4]; //线段树

void dfs1(int u,int f){ //搞fa,dep,siz,son
  fa[u]=f,dep[u]=dep[f]+1,siz[u]=1;
  for(int v:e[u]){
    if(v==f) continue;
    dfs1(v,u);
    siz[u]+=siz[v];
    if(siz[son[u]]<siz[v]) son[u]=v;
  }
}
void dfs2(int u,int tp){ //搞top,id,nw
  top[u]=tp,id[u]=++cnt,nw[cnt]=w[u];
  if(son[u]) dfs2(son[u],tp);
  for(int v:e[u]){
    if(v==fa[u]||v==son[u])continue;
    dfs2(v,v);
  }
}

void pushup(int u){
  tr[u].sum=tr[lc].sum+tr[rc].sum;
}
void pushdown(int u){
  if(tr[u].add){
    tr[lc].sum+=tr[u].add*(tr[lc].r-tr[lc].l+1);
    tr[rc].sum+=tr[u].add*(tr[rc].r-tr[rc].l+1);
    tr[lc].add+=tr[u].add;
    tr[rc].add+=tr[u].add;
    tr[u].add=0;
  }
}
void build(int u,int l,int r){ //构建线段树
  tr[u]={l,r,nw[l],0};
  if(l==r) return;
  int mid=l+r>>1;
  build(lc,l,mid);
  build(rc,mid+1,r);
  pushup(u);
}
void change(int u,int x,int y,int k){ //线段树修改
  if(x>tr[u].r||y<tr[u].l) return;
  if(x<=tr[u].l&&tr[u].r<=y){
    tr[u].sum+=k*(tr[u].r-tr[u].l+1);
    tr[u].add+=k;
    return;
  }
  pushdown(u);
  change(lc,x,y,k);
  change(rc,x,y,k);
  pushup(u);
}
void change_path(int u,int v,int k){ //修改路径
  while(top[u]!=top[v]){
    if(dep[top[u]]<dep[top[v]]) swap(u,v);
    change(1,id[top[u]],id[u],k);
    u=fa[top[u]];
  }
  if(dep[u]<dep[v]) swap(u,v);
  change(1,id[v],id[u],k); //最后一段
}
void change_tree(int u,int k){ //修改子树
  change(1,id[u],id[u]+siz[u]-1,k);
}
int query(int u,int x,int y){ //线段树查询
  if(x>tr[u].r||y<tr[u].l) return 0;
  if(x<=tr[u].l&&tr[u].r<=y) return tr[u].sum;
  pushdown(u);
  return query(lc,x,y)+query(rc,x,y);
}
int query_path(int u,int v){ //查询路径
  int res=0;
  while(top[u]!=top[v]){
    if(dep[top[u]]<dep[top[v]]) swap(u,v);
    res+=query(1,id[top[u]],id[u]);
    u=fa[top[u]];
  }
  if(dep[u]<dep[v]) swap(u,v);
  res+=query(1,id[v],id[u]); //最后一段
  return res;
}
int query_tree(int u){ //查询子树
  return query(1,id[u],id[u]+siz[u]-1);
}
signed main(){
  ios::sync_with_stdio(0),cin.tie(0),cout.tie(0);
  cin>>n>>m>>root>>P;
  for(int i=1; i<=n; i++) cin>>w[i];
  for(int i=0; i<n-1; i++){
    cin>>a>>b;
    e[a].push_back(b); e[b].push_back(a);
  }
  dfs1(root,0); dfs2(root,root); //把树拆成链
  build(1,1,n);  //用链建线段树
  while(m--){
    cin>>t>>a;
    if(t==1) cin>>b>>c,change_path(a,b,c);
    else if(t==3) cin>>c,change_tree(a,c);
    else if(t==2) cin>>b,cout<<query_path(a,b)%P<<"\n";
    else cout<<query_tree(a)%P<<"\n";
  }
}