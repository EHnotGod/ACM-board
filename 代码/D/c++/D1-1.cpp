// Kahn算法 O(n)
#include <iostream>
#include <cstring>
#include <algorithm>
#include <queue>
using namespace std;
const int N = 100010;
int n,m,a,b;
vector<int> e[N], tp;
int din[N];

bool toposort(){
  queue<int> q;
  for(int i = 1; i <= n; i++)
    if(din[i]==0) q.push(i);
  while(q.size()){
    int x=q.front(); q.pop();
    tp.push_back(x);
    for(auto y : e[x]){
      if(--din[y]==0) q.push(y);
    }
  }
  return tp.size() == n;
}
int main(){
  cin >> n >> m;
  for(int i=0; i<m; i++){
    cin >> a >> b;
    e[a].push_back(b);
    din[b]++;
  }
  if(!toposort()) puts("-1");
  else for(auto x:tp)printf("%d ",x);
  return 0;
}