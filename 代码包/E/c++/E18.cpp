// 树上背包 O(n^2)
#include <iostream>
#include <cstring>
#include <algorithm>
#include <vector>
using namespace std;

const int N=305;
vector<int> e[N];
int n,m,w[N],f[N][N],siz[N];

void dfs(int u){
  f[u][1]=w[u];siz[u]=1;
  for(int v:e[u]){
    dfs(v);
    siz[u]+=siz[v];
    for(int j=min(m+1,siz[u]);j;j--) //课程
      for(int k=0;k<=min(j-1,siz[v]);k++) //决策
        f[u][j]=max(f[u][j],f[u][j-k]+f[v][k]);
  }
}
int main(){
  scanf("%d%d",&n,&m);
  for(int i=1,k; i<=n; i++){
    scanf("%d%d",&k,&w[i]);
    e[k].push_back(i);
  }
  dfs(0); //虚拟根节点0
  printf("%d",f[0][m+1]);
}