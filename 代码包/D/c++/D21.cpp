// Luogu P3376 【模板】网络最大流
#include <iostream>
#include <cstring>
#include <algorithm>
#include <queue>
#define LL long long
#define N 10010
#define M 200010
using namespace std;

int n,m,S,T;
struct edge{LL v,c,ne;}e[M];
int h[N],idx=1; //从2,3开始配对
int d[N],cur[N];

void add(int a,int b,int c){
  e[++idx]={b,c,h[a]};
  h[a]=idx;
}
bool bfs(){ //对点分层，找增广路
  memset(d,0,sizeof d);
  queue<int>q;
  q.push(S); d[S]=1;
  while(q.size()){
    int u=q.front(); q.pop();
    for(int i=h[u];i;i=e[i].ne){
      int v=e[i].v;
      if(d[v]==0 && e[i].c){
        d[v]=d[u]+1;
        q.push(v);
        if(v==T)return true;
      }
    }
  }
  return false;
}
LL dfs(int u, LL mf){ //多路增广
  if(u==T) return mf;
  LL sum=0;
  for(int i=cur[u];i;i=e[i].ne){
    cur[u]=i; //当前弧优化
    int v=e[i].v;
    if(d[v]==d[u]+1 && e[i].c){
      LL f=dfs(v,min(mf,e[i].c));
      e[i].c-=f;
      e[i^1].c+=f; //更新残留网
      sum+=f; //累加u的流出流量
      mf-=f;  //减少u的剩余流量
      if(mf==0)break;//余量优化
    }
  }
  if(sum==0) d[u]=0; //残枝优化
  return sum;
}
LL dinic(){ //累加可行流
  LL flow=0;
  while(bfs()){
    memcpy(cur, h, sizeof h);
    flow+=dfs(S,1e9);
  }
  return flow;
}
int main(){
  int a,b,c;
  scanf("%d%d%d%d",&n,&m,&S,&T);
  while(m -- ){
    scanf("%d%d%d",&a,&b,&c);
    add(a,b,c); add(b,a,0);
  }
  printf("%lld\n",dinic());
  return 0;
}