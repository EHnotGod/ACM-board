// Tarjan算法 O(n+m)
#include<bits/stdc++.h>
using namespace std;

const int N=10010;
int n,m,a,b,ans;
vector<int> e[N];
int dfn[N],low[N],tim,stk[N],ins[N],top,scc[N],siz[N],cnt;

void tarjan(int x){
  dfn[x]=low[x]=++tim; //时间戳 追溯值
  stk[++top]=x,ins[x]=1;
  for(int y:e[x]){
    if(!dfn[y]){ //若y尚未访问
      tarjan(y);
      low[x]=min(low[x],low[y]); //因y是儿子
    }
    else if(ins[y]) //若y已访问且在栈中
      low[x]=min(low[x],dfn[y]); //因y是祖先或左子树点
  }

  if(dfn[x]==low[x]){ //若x是SCC的根
    ++cnt;
    while(1){
      int y=stk[top--]; ins[y]=0;
      scc[y]=cnt; //SCC的编号
      ++siz[cnt]; //SCC的大小
      if(y==x) break;
    }
  }
}
int main(){
  ios::sync_with_stdio(0),cin.tie(0),cout.tie(0);
  cin>>n>>m;
  while(m--)
    cin>>a>>b, e[a].push_back(b);
  for(int i=1;i<=n;i++) //可能不连通
    if(!dfn[i]) tarjan(i);
  for(int i=1;i<=cnt;i++)
     if(siz[i]>1) ans++;
  cout<<ans;
}