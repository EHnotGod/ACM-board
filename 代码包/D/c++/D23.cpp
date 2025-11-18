// Luogu P3381 【模板】最小费用最大流
#include <iostream>
#include <cstring>
#include <algorithm>
#include <queue>
using namespace std;

const int N=5010,M=100010,INF=1e8;
int n,m,S,T;
struct edge{int v,c,w,ne;}e[M];
int h[N],idx=1;//从2,3开始配对
int d[N],mf[N],pre[N],vis[N];
int flow,cost;

void add(int a,int b,int c,int d){
  e[++idx]={b,c,d,h[a]};
  h[a]=idx;
}
bool spfa(){
  memset(d,0x3f,sizeof d);
  memset(mf,0,sizeof mf);
  queue<int> q; q.push(S);
  d[S]=0, mf[S]=INF, vis[S]=1;
  while(q.size()){
    int u=q.front(); q.pop();
    vis[u]=0;
    for(int i=h[u];i;i=e[i].ne){
      int v=e[i].v,c=e[i].c,w=e[i].w;
      if(d[v]>d[u]+w && c){
        d[v]=d[u]+w; //最短路
        pre[v]=i;
        mf[v]=min(mf[u],c);
        if(!vis[v]){
          q.push(v); vis[v]=1;
        }
      }
    }
  }
  return mf[T]>0;
}
void EK(){
  while(spfa()){
    for(int v=T;v!=S;){
      int i=pre[v];
      e[i].c-=mf[T];
      e[i^1].c+=mf[T];
      v=e[i^1].v;
    }
    flow+=mf[T]; //累加可行流
    cost+=mf[T]*d[T];//累加费用
  }
}
int main(){
  scanf("%d%d%d%d",&n,&m,&S,&T);
  int a,b,c,d;
  while(m --){
    scanf("%d%d%d%d",&a,&b,&c,&d);
    add(a,b,c,d);
    add(b,a,0,-d);
  }
  EK();
  printf("%d %d\n",flow,cost);
  return 0;
}