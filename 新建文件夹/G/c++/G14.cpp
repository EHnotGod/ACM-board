#include <iostream>
using namespace std;

typedef long long LL;
int a,b,m,phi,flag;
char s[20000005];

int get_phi(int n){//求欧拉函数
  int res = n;
  for(int i=2; i*i<=n; i++){
    if(n%i == 0){
      res = res/i*(i-1);
      while(n%i == 0) n /= i;
    }
  }
  if(n>1) res = res/n*(n-1);
  return res;
}
int depow(int phi){//降幂
  int b = 0;
    for(int i=0; s[i]; i++){
      b = b*10+(s[i]-'0');
      if(b>=phi) flag=1, b%=phi;
    }
    if(flag) b += phi;
    return b;
}
int qpow(LL a, int b){//快速幂
    int res = 1;
    while(b){
      if(b&1) res = res*a%m;
      a = a*a%m;
      b >>= 1;
    }
    return res;
}
int main(){
    scanf("%d%d%s", &a, &m, s);
    phi = get_phi(m);
  b = depow(phi);
    printf("%d", qpow(a,b));
    return 0;
}