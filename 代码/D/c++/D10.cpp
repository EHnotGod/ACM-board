// Tarjan算法 O(n+m)
#include<bits/stdc++.h>
using namespace std;

const int N=500005,M=2*N;
int n,m,s,a,b;
vector<int> e[N];
vector<pair<int,int>> query[N];
int fa[N],vis[N],ans[M];

int find(int x){
  if(x==fa[x]) return x;
  return fa[x]=find(fa[x]);
}
void tarjan(int x){
  vis[x]=true; //标记x已访问
  for(auto y:e[x]){
    if(!vis[y]){
      tarjan(y);
      fa[y]=x; //回到x时指向x
    }
  }
  for(auto q : query[x]){ //离开x时找LCA
    int y=q.first,i=q.second;
    if(vis[y])ans[i]=find(y);
  }
}
int main(){
  scanf("%d%d%d",&n,&m,&s);
  for(int i=1; i<n; i++){
    scanf("%d%d",&a,&b);
    e[a].push_back(b);
    e[b].push_back(a);
  }
  for(int i=1;i<=m;i++){
    scanf("%d%d",&a,&b);
    query[a].push_back({b,i});
    query[b].push_back({a,i});
  }
  for(int i=1;i<=N;i++)fa[i]=i;
  tarjan(s);
  for(int i=1; i<=m; i++)
    printf("%d\n",ans[i]);
}