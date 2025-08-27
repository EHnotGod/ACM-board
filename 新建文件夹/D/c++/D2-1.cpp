//Dijkstra
#include <iostream>
#include <cstring>
#include <algorithm>
#include <vector>
#define inf 2147483647
using namespace std;

int n,m,s,a,b,c;
const int N=100010;
struct edge{int v,w;};
vector<edge> e[N];
int d[N], vis[N];

void dijkstra(int s){
  for(int i=0;i<=n;i++)d[i]=inf;
  d[s]=0;
  for(int i=1;i<n;i++){//枚举次数
    int u=0;
    for(int j=1;j<=n;j++)//枚举点
      if(!vis[j]&&d[j]<d[u]) u=j;
    vis[u]=1; //标记u已出圈
    for(auto ed:e[u]){//枚举邻边
      int v=ed.v, w=ed.w;
      if(d[v]>d[u]+w){
        d[v]=d[u]+w;
      }
    }
  }
}
int main(){
  cin>>n>>m>>s;
  for(int i=0; i<m; i++){
    cin>>a>>b>>c;
    e[a].push_back({b,c});
  }
  dijkstra(s);
  for(int i=1;i<=n;i++)
    printf("%d ",d[i]);
  return 0;
}