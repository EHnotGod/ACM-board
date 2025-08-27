// 逆序枚举，优化空间#include <iostream>
#include <cstring>
#include <algorithm>
using namespace std;

const int N=3410,M=13000;
int n, m;
int v[N],w[N],f[M];

int main(){
  scanf("%d%d",&n,&m);
  for(int i=1; i<=n; i++)
    scanf("%d%d",&v[i],&w[i]);  //费用，价值

  for(int i=1; i<=n; i++)       //枚举物品
    for(int j=m; j>=v[i]; j--)  //枚举体积
      f[j]=max(f[j],f[j-v[i]]+w[i]);

  printf("%d\n",f[m]);
}