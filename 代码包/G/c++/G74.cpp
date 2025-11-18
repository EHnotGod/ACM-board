// 拉格朗日插值法 O(n^2)
#include <iostream>
#include <cstring>
#include <algorithm>
using namespace std;

#define LL long long
const LL mod=998244353;
LL n,k,ans;
LL x[2005],y[2005];

LL ksm(LL a,LL b){
  LL s=1;
  while(b){
    if(b&1)s=s*a%mod;
    a=a*a%mod;
    b>>=1;
  }
  return s;
}
int main(){
  cin>>n>>k;
  for(int i=1;i<=n;i++)cin>>x[i]>>y[i];
  for(int i=1;i<=n;i++){
    LL a=y[i],b=1;
    for(int j=1;j<=n;j++){
      if(i==j) continue;
      a=a*(k-x[j])%mod;
      b=b*(x[i]-x[j])%mod;
    }
    ans=(ans+a*ksm(b,mod-2)%mod)%mod;
  }
  cout<<(ans+mod)%mod;
}