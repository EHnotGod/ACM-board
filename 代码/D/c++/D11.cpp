// 树链剖分 O(mlogn)
#include <iostream>
#include <cstring>
#include <algorithm>
#include <vector>
using namespace std;

const int N=500010;
int n,m,s,a,b;
vector<int> e[N];
int fa[N],son[N],dep[N],siz[N],top[N];

void dfs1(int u,int f){ //搞fa,son,dep
  fa[u]=f;siz[u]=1;dep[u]=dep[f]+1;
  for(int v:e[u]){
    if(v==f) continue;
    dfs1(v,u);
    siz[u]+=siz[v];
    if(siz[son[u]]<siz[v])son[u]=v;
  }
}
void dfs2(int u,int t){ //搞top
  top[u]=t;  //记录链头
  if(!son[u]) return; //无重儿子
  dfs2(son[u],t);     //搜重儿子
  for(int v:e[u]){
    if(v==fa[u]||v==son[u])continue;
    dfs2(v,v); //搜轻儿子
  }
}
int lca(int u,int v){
  while(top[u]!=top[v]){
    if(dep[top[u]]<dep[top[v]])swap(u,v);
    u=fa[top[u]];
  }
  return dep[u]<dep[v]?u:v;
}
int main(){
  scanf("%d%d%d",&n,&m,&s);
  for(int i=1; i<n; i++){
    scanf("%d%d",&a,&b);
    e[a].push_back(b);
    e[b].push_back(a);
  }
  dfs1(s,0);
  dfs2(s,s);
  while(m--){
    scanf("%d%d",&a,&b);
    printf("%d\n",lca(a,b));
  }
  return 0;
}