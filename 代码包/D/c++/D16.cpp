#include<bits/stdc++.h>
using namespace std;

const int N=20010;
int n,m,a,b;
vector<int> e[N];
int dfn[N],low[N],tim,cut[N],root;

void tarjan(int x){
  dfn[x]=low[x]=++tim;
  int son=0; //x的儿子个数
  for(int y:e[x]){
    if(!dfn[y]){ //若y未访问
      tarjan(y);
      low[x]=min(low[x],low[y]);
      if(low[y]>=dfn[x]){
        son++;
        if(x!=root||son>1) cut[x]=1;
      }
    }
    else //若y已访问
      low[x]=min(low[x],dfn[y]); //注:dfn不能换成low
  }
}
int main(){
  cin>>n>>m;
  while(m --){
    cin>>a>>b;
    e[a].push_back(b),
    e[b].push_back(a);
  }
  for(root=1;root<=n;root++) if(!dfn[root]) tarjan(root);

  int ans=0;
  for(int i=1;i<=n;i++) if(cut[i]) ans++;
  cout<<ans<<"\n";
  for(int i=1;i<=n;i++) if(cut[i]) cout<<i<<" ";
}