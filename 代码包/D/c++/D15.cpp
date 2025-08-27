#include<bits/stdc++.h>
using namespace std;

const int N=100010;
int n,m,a,b;
vector<int> e[N],ne[N];
int dfn[N],low[N],tim,stk[N],top,scc[N],cnt;
int w[N],nw[N],d[N];

void tarjan(int x){
  dfn[x]=low[x]=++tim;
  stk[++top]=x;
  for(int y : e[x]){
    if(!dfn[y]){
      tarjan(y);
      low[x]=min(low[x],low[y]);
    }
    else if(!scc[y])
      low[x]=min(low[x],dfn[y]);
  }
  if(dfn[x]==low[x]){
    ++cnt;
    while(1){
      int y=stk[top--];
      scc[y]=cnt;
      if(y==x) break;
    }
  }
}
int main(){
  cin>>n>>m;
  for(int i=1;i<=n;i++) cin>>w[i];
  for(int i=1;i<=m;i++){
    cin>>a>>b;
    e[a].push_back(b);
  }

  for(int i=1;i<=n;i++) //缩点
    if(!dfn[i]) tarjan(i);
  for(int x=1;x<=n;x++){ //建拓扑图
    nw[scc[x]]+=w[x];
    for(int y:e[x])
      if(scc[x]!=scc[y]) ne[scc[x]].push_back(scc[y]);
  }
  for(int x=cnt;x;x--){ //求最长路
    if(d[x]==0) d[x]=nw[x]; //起点
    for(int y:ne[x])
      d[y]=max(d[y],d[x]+nw[y]);
  }
  int ans=0;
  for(int i=1;i<=cnt;i++) ans=max(ans,d[i]);
  cout<<ans;
}