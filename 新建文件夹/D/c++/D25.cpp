// Luogu P3386 【模板】二分图最大匹配
#include <iostream>
#include <cstring>
#include <algorithm>
using namespace std;

const int N=505,M=100010;
int n,m,k,a,b,ans;
struct edge{int v,ne;}e[M];
int h[N],idx;
int vis[N],match[N];

void add(int a,int b){
  e[++idx]={b,h[a]};
  h[a]=idx;
}
bool dfs(int u){
  for(int i=h[u];i;i=e[i].ne){
    int v=e[i].v; //妹子
    if(vis[v]) continue;
    vis[v]=1; //先标记这个妹子
    if(!match[v]||dfs(match[v])){
      match[v]=u; //配成对
      return 1;
    }
  }
  return 0;
}
int main(){
  cin>>n>>m>>k;
  for(int i=0;i<k;i++)
    cin>>a>>b, add(a,b);
  for(int i=1;i<=n;i++){
    memset(vis,0,sizeof vis);
    if(dfs(i)) ans++;
  }
  cout<<ans;
  return 0;
}