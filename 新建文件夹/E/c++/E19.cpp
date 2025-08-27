// 不超背包容量的方案数
#include<iostream>
#include<cstring>
using namespace std;

const int N=1010, mod=1e9+7;
int f[N],c[N];
// f[i]表示背包容量为i时最优选法的总价值
// c[i]表示背包容量为i时最优选法的方案数

int main(){
  int n, m, v, w;
  scanf("%d%d", &n, &m);
  for(int i=0;i<=m;i++) c[i]=1;

  for(int i=1; i<=n; i++){    //枚举物品
    scanf("%d%d",&v,&w);
    for(int j=m; j>=v; j--){  //枚举体积
      if(f[j-v]+w>f[j]){      //装新物品总价值更大
        f[j]=f[j-v]+w;
        c[j]=c[j-v];
      }
      else if(f[j-v]+w==f[j]) //装新物品总价值相等
        c[j]=(c[j]+c[j-v])%mod;
    }
  }
  printf("%d\n",c[m]);
}