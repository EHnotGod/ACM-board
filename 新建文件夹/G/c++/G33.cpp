#include<cstdio>
#include<algorithm>
using namespace std;

typedef long long LL;

int main(){
  LL n, k, l, r, res;
  scanf("%lld%lld", &n, &k);
  res = n*k;
  for(l=1; l<=n; l=r+1){
    if(k/l == 0) break;
    r = min(k/(k/l),n);
    res -= (k/l)*(r-l+1)*(l+r)/2;
  }
  printf("%lld", res);
  return 0;
}