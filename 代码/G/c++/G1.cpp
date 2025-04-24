#include <iostream>
using namespace std;

typedef long long LL;
int a,b,p;

int qpow(int a,int b,int p){ //快速幂
  int s=1;
  while(b){
    if(b&1) s=(LL)s*a%p;
    a=(LL)a*a%p;
    b>>=1;
  }
  return s;
}
int main(){
  cin>>a>>b>>p;
  int s=qpow(a,b,p);
  printf("%d^%d mod %d=%d\n",a,b,p,s);
  return 0;
}