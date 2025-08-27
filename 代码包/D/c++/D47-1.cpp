// 树的直径 正边权 两次DFS O(n)
#include<bits/stdc++.h>
using namespace std;

const int N=100005;
int n,rt,d[N];
vector<pair<int,int>> e[N];

void dfs(int u,int fa){
  if(d[rt]<d[u]) rt=u; //记录最远点
  for(auto [v,w]:e[u]){
    if(v==fa) continue;
    d[v]=d[u]+w; //d[v]从根走到v的距离
    dfs(v,u);
  }
}
int main(){
  cin>>n;
  for(int i=1,x,y;i<n;i++){
    cin>>x>>y;
    e[x].emplace_back(y,1);
    e[y].emplace_back(x,1);
  }
  dfs(1,0);  //找出离1最远的点rt
  d[rt]=0;
  dfs(rt,0); //找出离rt最远的点
  cout<<d[rt];
}