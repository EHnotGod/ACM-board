// 线性基 O(63*n)
#include <iostream>
#include <cstring>
#include <algorithm>
using namespace std;

typedef long long LL;
int n,k;
LL p[64];

void gauss(){ //高斯消元法
  for(int i=63;i>=0;i--){
    // 把当前第i位是1的数换上去
    for(int j=k;j<n;j++)
      if(p[j]>>i&1){swap(p[j],p[k]); break;}
    // 当前第i位所有向量都是0
    if((p[k]>>i&1)==0) continue;
    // 把其他数的第i位全部消为0
    for(int j=0;j<n;j++)
      if(j!=k&&(p[j]>>i&1)) p[j]^=p[k];
    // 基的个数+1
    k++; if(k==n) break;
  }
}
int main(){
  scanf("%d",&n);
  for(int i=0;i<n;i++)scanf("%lld",&p[i]);
  gauss();
  LL ans=0;
  for(int i=0;i<k;i++) ans^=p[i];
  printf("%lld\n",ans);
}