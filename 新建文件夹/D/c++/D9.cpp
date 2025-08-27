// 倍增法 O(nlogn)
#include<bits/stdc++.h>
using namespace std;

const int N=500005;
int n,m,s;
vector<int> e[N];
int f[N][22],dep[N];

void dfs(int u,int fa){
  f[u][0]=fa; dep[u]=dep[fa]+1;
  for(int i=1;i<=20;i++) //u的2,4,8...祖先
    f[u][i]=f[f[u][i-1]][i-1];
  for(int v:e[u])
    if(v!=fa) dfs(v,u);
}
int lca(int u,int v){
  if(dep[u]<dep[v]) swap(u,v);
  for(int i=20;~i;i--) //u先大步后小步向上跳，直到与v同层
    if(dep[f[u][i]]>=dep[v]) u=f[u][i];
  if(u==v) return v;
  for(int i=20;~i;i--) //u,v一起向上跳，直到lca的下面
    if(f[u][i]!=f[v][i]) u=f[u][i],v=f[v][i];
  return f[u][0];
}
int main(){
  scanf("%d%d%d",&n,&m,&s);
  for(int i=1,a,b; i<n; i++){
    scanf("%d%d",&a,&b);
    e[a].push_back(b);
    e[b].push_back(a);
  }
  dfs(s,0);
  for(int i=0,a,b;i<m;i++){
    scanf("%d%d",&a,&b);
    printf("%d\n",lca(a,b));
  }
}