#include <iostream>
#include <cstring>
#include <algorithm>
#include <vector>
#include <cmath>
using namespace std;

const int N=500;
typedef vector<int> VI;
VI a(N),res(N);
int p;

VI mul(VI &a, VI &b){//高精度
  VI t(N*2);
  for(int i=0; i<N; i++)
    for(int j=0; j<N; j++){
      t[i+j] += a[i]*b[j];
      t[i+j+1] += t[i+j]/10;
      t[i+j] %= 10;
    }
  return t;
}
void quickpow(int p){//快速幂
  res[0]=1, a[0]=2;
  while(p){
    if(p & 1) res = mul(res,a);
    a = mul(a,a);
    p >>= 1;
  }
  res[0]--; //个位修正
}
int main(){
  cin >> p;
  printf("%d\n",int(p*log10(2))+1);
  quickpow(p);
  for(int i=0, k=499; i<10; i++){
    for(int j=0; j<50; j++, k--)
      printf("%d",res[k]);
    puts("");
  }
  return 0;
}