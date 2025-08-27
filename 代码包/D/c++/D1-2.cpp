// DFS算法 O(n)
#include <iostream>
#include <cstring>
#include <algorithm>
#include <queue>
using namespace std;

const int N = 100010;
int n,m,a,b;
vector<int> e[N], tp;
int c[N]; //染色数组

bool dfs(int x){
  c[x] = -1;
  for(int y : e[x]){
    if(c[y]<0)return 0; //有环
    else if(!c[y])
      if(!dfs(y))return 0;
  }
  c[x] = 1;
  tp.push_back(x);
  return 1;
}
bool toposort(){
  memset(c, 0, sizeof(c));
  for(int x = 1; x <= n; x++)
    if(!c[x])
      if(!dfs(x))return 0;
  reverse(tp.begin(),tp.end());
  return 1;
}
int main(){
  cin >> n >> m;
  for(int i=0; i<m; i++){
    cin >> a >> b;
    e[a].push_back(b);
  }
  if(!toposort()) puts("-1");
  else
    for(int x:tp)printf("%d ",x);
  return 0;
}