#include<iostream>
#include<cstdio>
#include<cmath>
#define LL long long
using namespace std;

const int N=405,P=1e9+7;
int n;
LL a[N][N<<1];

LL quickpow(LL a, LL b){
  LL ans = 1;
  while(b){
    if(b & 1) ans = ans*a%P;
    a = a*a%P;
    b >>= 1;
  }
  return ans;
}
bool Gauss_Jordan(){
  for(int i=1;i<=n;++i){ //枚举主元的行列
    int r = i;
    for(int k=i; k<=n; ++k) //找非0行
      if(a[k][i]) {r=k; break;}
    if(r!=i) swap(a[r],a[i]); //换行
    if(!a[i][i]) return 0;

    int x=quickpow(a[i][i],P-2); //求逆元
    for(int k=1; k<=n; ++k){ //对角化
      if(k == i) continue;
      int t=a[k][i]*x%P;
      for(int j=i; j<=2*n; ++j)
        a[k][j]=((a[k][j]-t*a[i][j])%P+P)%P;
    }
    for(int j=1; j<=2*n; ++j) //除以主元
      a[i][j]=(a[i][j]*x%P);
  }
  return 1;
}
int main(){
  scanf("%d",&n);
  for(int i=1; i<=n; ++i)
    for(int j=1; j<=n; ++j)
      scanf("%lld",&a[i][j]),a[i][i+n]=1;
  if(Gauss_Jordan())
    for(int i=1; i<=n; ++i){
      for(int j=n+1; j<=2*n; ++j)
        printf("%lld ",a[i][j]);
      puts("");
    }
  else puts("No Solution");
  return 0;
}