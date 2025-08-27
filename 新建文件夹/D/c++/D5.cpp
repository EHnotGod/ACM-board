#include<algorithm>
#include<cstring>
#include<iostream>
#include<queue>
#define N 30010
#define INF 1000000000
using namespace std;

int n,m,a,b,c;
struct edge{int v,w;};
vector<edge> e[N];
int vis[N],cnt[N];
long long h[N],d[N];

void spfa(){
    queue<int>q;
    memset(h,63,sizeof h);
    memset(vis,false,sizeof vis);
    h[0]=0,vis[0]=1;q.push(0);
    while(q.size()){
        int u=q.front(); q.pop();vis[u]=0;
        for(auto ed : e[u]){
            int v=ed.v,w=ed.w;
            if(h[v]>h[u]+w){
                h[v]=h[u]+w;
        cnt[v]=cnt[u]+1;
        if(cnt[v]>n){
          printf("-1\n");exit(0);
        }
                if(!vis[v])q.push(v),vis[v]=1;
            }
        }
    }
}
void dijkstra(int s){
    priority_queue<pair<long long,int>>q;
    for(int i=1;i<=n;i++)d[i]=INF;
    memset(vis,false,sizeof vis);
    d[s]=0; q.push({0,s});
    while(q.size()){
        int u=q.top().second;q.pop();
        if(vis[u])continue;
        vis[u]=1;
        for(auto ed : e[u]){
            int v=ed.v,w=ed.w;
            if(d[v]>d[u]+w){
                d[v]=d[u]+w;
                if(!vis[v]) q.push({-d[v],v});
            }
        }
    }
}
int main(){
  cin>>n>>m;
  for(int i=0;i<m;i++)
    cin>>a>>b>>c, e[a].push_back({b,c});
    for(int i=1;i<=n;i++)
      e[0].push_back({i,0});//加虚拟边

    spfa();
    for(int u=1;u<=n;u++)
      for(auto &ed:e[u])
        ed.w+=h[u]-h[ed.v];//构造新边
    for(int i=1;i<=n;i++){
        dijkstra(i);
        long long ans=0;
        for(int j=1;j<=n;j++){
            if(d[j]==INF) ans+=(long long)j*INF;
            else ans+=(long long)j*(d[j]+h[j]-h[i]);
        }
        printf("%lld\n",ans);
    }
    return 0;
}