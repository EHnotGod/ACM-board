// 分组背包 朴素算法
#include<iostream>
#include<cstring>
using namespace std;

const int N=110;
int v[N][N],w[N][N],s[N];
// v[i,j]:第i组第j个物品的体积 s[i]:第i组物品的个数
int f[N][N];
// f[i,j]:前i组物品，能放入容量为j的背包的最大值

int main(){
  int n,V; cin>>n>>V;
  for(int i=1;i<=n;i++){
    cin>>s[i];
    for(int j=1;j<=s[i];j++) cin>>v[i][j]>>w[i][j];
  }

  for(int i=1;i<=n;i++)     //物品组
  for(int j=1;j<=V;j++)     //体积
  for(int k=0;k<=s[i];k++)  //同组内的物品只能选一个
    if(j>=v[i][k]) f[i][j]=max(f[i][j],f[i-1][j-v[i][k]]+w[i][k]);

  cout<<f[n][V];
}