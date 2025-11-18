#include<bits/stdc++.h>
using namespace std;

const int N=500010,M=4000010;
int n,m,a,b;
int h[N],to[M],ne[M],idx=1;
void add(int a,int b){
  to[++idx]=b,ne[idx]=h[a],h[a]=idx;
}
int dfn[N],low[N],tim,stk[N],top,dcc[N],siz[N],cnt,bri[M];
vector<int> d[N];

void tarjan(int x,int ine){
  dfn[x]=low[x]=++tim;stk[++top]=x;
  for(int i=h[x];i;i=ne[i]){
    int y=to[i];
    if(!dfn[y]){
      tarjan(y,i);
      low[x]=min(low[x],low[y]);
      if(low[y]<dfn[x]) bri[i]=bri[i^1]=1;
    }
    else if(i!=(ine^1)) low[x]=min(low[x],dfn[y]);
  }

  if(dfn[x]==low[x]){
    ++cnt;
    while(1){
      int y=stk[top--];
      // dcc[y]=cnt;
      siz[cnt]++;
      d[cnt].push_back(y);
      if(x==y) break;
    }
  }
}
int main(){
  ios::sync_with_stdio(0),cin.tie(0),cout.tie(0);
  cin>>n>>m;
  while(m--){
    cin>>a>>b;add(a,b);add(b,a);
  }
  for(int i=1;i<=n;i++)if(!dfn[i])tarjan(i,0);
  cout<<cnt<<"\n";
  for(int i=1;i<=cnt;i++){
    cout<<siz[i]<<" ";
    for(int j:d[i]) cout<<j<<" ";
    cout<<"\n";
  }
}