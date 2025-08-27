// 树的直径 正负边权 树形DP O(n)
#include<bits/stdc++.h>
using namespace std;

const int N=100005;
int n,mxd,d[N]; //d[u]从u点向下走的最长距离
vector<pair<int,int>> e[N];

void dfs(int u,int fa){
  for(auto [v,w]:e[u]){
    if(v==fa) continue;
    dfs(v,u);
    mxd=max(mxd,d[u]+w+d[v]); //拼凑直径
    d[u]=max(d[u],d[v]+w);    //更新d[u]
  }
}
int main(){
  cin>>n;
  for(int i=1,x,y;i<n;i++){
    cin>>x>>y;
    e[x].emplace_back(y,1);
    e[y].emplace_back(x,1);
  }
  dfs(1,0);
  cout<<mxd;
}