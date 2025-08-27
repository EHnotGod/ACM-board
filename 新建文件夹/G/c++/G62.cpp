// 线性基 O(63*n)
#include <iostream>
#include <cstring>
#include <algorithm>
using namespace std;

typedef long long LL;
const int N=10005;
int T,n,m,s;
LL p[N];

void gauss(){
  s=0;
  for(int i=63;i>=0;i--){
    // 把当前第i位是1的数换上去
    for(int j=s;j<n;j++)
      if(p[j]>>i&1){swap(p[j],p[s]);break;}
    // 当前第i位所有向量都是0
    if((p[s]>>i&1)==0) continue;
    // 把其他数的第i位全部消为0
    for(int j=0;j<n;j++)
      if(j!=s&&(p[j]>>i&1)) p[j]^=p[s];
    // 有多组测试数据，不break，会被上一组数据影响
    s++; if(s==n) break;
  }
}
int main(){
  scanf("%d",&T);
  for(int C=1;C<=T;C++){
    printf("Case #%d:\n",C);
    scanf("%d",&n);
    for(int i=0;i<n;i++)scanf("%lld",&p[i]);
    gauss(); //高斯消元法构造线性基
    scanf("%d",&m);
    while(m--){
      LL k; scanf("%lld",&k); //第k小
      if(s<n) k--;      //如果能凑出0
      if(k>=(1ll<<s)) puts("-1");
      else{
        LL ans=0;
        for(int i=0;i<s;i++)
          if(k>>i&1) ans^=p[s-i-1];
        printf("%lld\n",ans);
      }
    }
  }
}