#include <cstring>
#include <iostream>
#include <algorithm>
#include <queue>
using namespace std;

const int N=100005;
int x, y, dis[N];

void bfs(){
  memset(dis,-1,sizeof dis); dis[x]=0;
  queue<int> q; q.push(x);
  while(q.size()){
    int x=q.front(); q.pop();
    if(x+1<N && dis[x+1]==-1){
      dis[x+1]=dis[x]+1; //前进一步
      q.push(x+1);
    }
    if(x-1>0 && dis[x-1]==-1){
      dis[x-1]=dis[x]+1; //后退一步
      q.push(x-1);
    }
    if(2*x<N && dis[2*x]==-1){
      dis[2*x]=dis[x]+1; //走到2x位置
      q.push(2*x);
    }
    if(x==y){printf("%d\n",dis[y]);return;}
  }
}
int main(){
  int T; cin>>T;
  while(T--) cin>>x>>y, bfs();
}