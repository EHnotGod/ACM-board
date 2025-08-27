//堆优化Dijkstra
#include <cstring>
#include <iostream>
#include <algorithm>
#include <queue>
using namespace std;

const int N=100010;
int n,m,s,a,b,c;
struct edge{int v,w;};
vector<edge> e[N];
int d[N],vis[N];

void dijkstra(int s){
  memset(d,0x3f,sizeof d); d[s]=0;
  priority_queue<pair<int,int>> q;
  q.push({0,s});
  while(q.size()){
    auto t=q.top(); q.pop();
    int u=t.second;
    if(vis[u])continue; //再出队跳过
    vis[u]=1; //标记u已出队
    for(auto ed : e[u]){
      int v=ed.v, w=ed.w;
      if(d[v]>d[u]+w){
        d[v]=d[u]+w;
        q.push({-d[v],v}); //大根堆
      }
    }
  }
}
int main(){
  cin>>n>>m>>s;
  for(int i=0; i<m; i++)
    cin>>a>>b>>c, e[a].push_back({b,c});
  dijkstra(s);
  for(int i=1;i<=n;i++) printf("%d ",d[i]);
}