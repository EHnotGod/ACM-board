// 树的重心 树形DP O(n)
#include<bits/stdc++.h>
using namespace std;

const int N=50010;
int n,siz[N],f[N],cnt=1e9;
vector<int> e[N],g;

void dfs(int u,int fa){
  siz[u]=1;
  for(auto v:e[u]){
    if(v==fa) continue;
    dfs(v,u);
    f[u]=max(f[u],siz[v]); //u的最大子树
    siz[u]+=siz[v];
  }
  f[u]=max(f[u],n-siz[u]); //删除u后的最大连通块
  cnt=min(cnt,f[u]);       //最大块最小化
}
int main(){
  scanf("%d",&n);
  for(int i=1,a,b;i<n;i++){
    scanf("%d%d",&a,&b);
    e[a].push_back(b);
    e[b].push_back(a);
  }
  dfs(1,0);
  for(int i=1;i<=n;i++) if(f[i]==cnt) g.push_back(i);
  for(int v:g) printf("%d ",v);
}