#include<bits/stdc++.h>
using namespace std;

const int N=500010;
int n,m,a,b;
vector<int> e[N],ne[N],dcc[N];
int dfn[N],low[N],tot,stk[N],top,cut[N],root,cnt;

void tarjan(int x){
  if(x==root&&!e[x].size()){ //孤立点
        dcc[++cnt].push_back(x);
        return;
    }
  dfn[x]=low[x]=++tot; stk[++top]=x;
  int son=0;
  for(int y:e[x]){
    if(!dfn[y]){ //若y未访问
      tarjan(y);
      low[x]=min(low[x],low[y]);
      if(low[y]>=dfn[x]){
        son++;
        if(x!=root||son>1)cut[x]=1; //割点

        ++cnt;
        while(1){
          int z=stk[top--];
          dcc[cnt].push_back(z);
          if(z==y) break; //让x留在栈中
        }
        dcc[cnt].push_back(x); //vDCC
      }
    }
    else //若y已访问
      low[x]=min(low[x],dfn[y]);
  }
}
int main(){
  ios::sync_with_stdio(0),cin.tie(0),cout.tie(0);
  cin>>n>>m;
  while(m--){
    cin>>a>>b;
    if(a==b) continue; //忽略自环
    e[a].push_back(b),
    e[b].push_back(a);
  }
  for(root=1;root<=n;root++)if(!dfn[root])tarjan(root);
  cout<<cnt<<"\n";
  for(int i=1;i<=cnt;i++){
    cout<<dcc[i].size()<<" ";
    for(int j:dcc[i])cout<<j<<" ";
    cout<<"\n";
  }
}