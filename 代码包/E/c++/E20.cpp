#include<iostream>
#include<cstring>
using namespace std;

const int N = 1010;
int v[N],w[N];
int f[N][N],p[N][N];

int main(){
  int n,m; cin>>n>>m;
  for(int i=1; i<=n; i++) cin>>v[i]>>w[i];

  for(int i=n; i>=1; i--)   //逆序取物
  for(int j=0; j<=m; j++){  //枚举体积
    f[i][j]=f[i+1][j];
    p[i][j]=j;              //记录路径列
    if(j>=v[i])
      f[i][j]=max(f[i][j],f[i+1][j-v[i]]+w[i]);
    if(j>=v[i] && f[i][j]==f[i+1][j-v[i]]+w[i])
      p[i][j]=j-v[i];
  }

  int j=m;
  for(int i=1; i<=n; i++)
    if(p[i][j]<j){
      printf("%d ",i);
      j=p[i][j];
    }
}