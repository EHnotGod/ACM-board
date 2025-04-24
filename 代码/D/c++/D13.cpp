#include <iostream>
#include <cstring>
#include <algorithm>
#include <vector>
using namespace std;
const int N=100010;
int n,m,a,b,c;
vector<int> e[N];
int dep[N],fa[N],son[N],sz[N],dis[N];
int top[N];

void dfs1(int u,int father){
  fa[u]=father,dep[u]=dep[father]+1,sz[u]=1;
  for(int v:e[u]){
    if(v==father) continue;
    dis[v]=dis[u]+1;
    dfs1(v,u);
    sz[u]+=sz[v];
    if(sz[son[u]]<sz[v])son[u]=v;
  }
}
void dfs2(int u,int t){
  top[u]=t;
  if(!son[u]) return;
  dfs2(son[u],t);
  for(int v:e[u]){
    if(v==fa[u]||v==son[u])continue;
    dfs2(v,v);
  }
}
int lca(int x,int y){
  while(top[x]!=top[y]){
    if(dep[top[x]]<dep[top[y]])swap(x,y);
    x=fa[top[x]];
  }
  return dep[x]<dep[y]?x:y;
}
int main(){
  scanf("%d",&n);
  for(int i=1; i<n; i++){
    scanf("%d%d",&a,&b);
    e[a].push_back(b);
    e[b].push_back(a);
  }
  dfs1(1,0);
  dfs2(1,1);
  scanf("%d",&m);
  while(m--){
    scanf("%d%d",&a,&b);
    int d=dis[a]+dis[b]-dis[lca(a,b)]*2;
    printf("%d\n",d);
  }
  return 0;
}