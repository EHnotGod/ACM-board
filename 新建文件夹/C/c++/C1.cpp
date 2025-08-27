//并查集 路径压缩
#include <iostream>
#include <cstring>
#include <algorithm>
using namespace std;

const int N=10005;
int n,m,x,y,z;
int pa[N];

int find(int x){
  if(pa[x]==x) return x;
  return pa[x]=find(pa[x]);
}
void unset(int x,int y){
  pa[find(x)]=find(y);
}
int main(){
  cin>>n>>m;
  for(int i=1;i<=n;i++) pa[i]=i;
  while(m --){
    cin>>z>>x>>y;
    if(z==1) unset(x,y);
    else{
      if(find(x)==find(y)) puts("Y");
      else puts("N");
    }
  }
}