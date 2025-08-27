#include <iostream>
#include <cstring>
#include <algorithm>
#include <cmath>
#include <unordered_map>
using namespace std;

typedef long long LL;

LL gcd(LL a, LL b){
  return b==0?a:gcd(b,a%b);
}
LL exbsgs(LL a, LL b, LL p){
  a %= p; b %= p;
  if(b==1||p==1)return 0;//x=0

  LL d, k=0, A=1;
  while(true){
    d = gcd(a,p);
    if(d==1) break;
    if(b%d) return -1; //无解
    k++; b/=d; p/=d;
    A = A*(a/d)%p; //求a^k/D
    if(A==b) return k;
  }

  LL m=ceil(sqrt(p));
  LL t = b;
  unordered_map<int,int> hash;
  hash[b] = 0;
  for(int j = 1; j < m; j++){
    t = t*a%p; //求b*a^j
    hash[t] = j;
  }
  LL mi = 1;
  for(int i = 1; i <= m; i++)
    mi = mi*a%p; //求a^m
  t = A;
  for(int i=1; i <= m; i++){
    t = t*mi%p; //求(a^m)^i
    if(hash.count(t))
      return i*m-hash[t]+k;
  }
  return -1; //无解
}
int main(){
  LL a, p, b;
  while((scanf("%lld%lld%lld",&a,&p,&b)!=EOF)&&a){
    LL res = exbsgs(a, b, p);
    if(res == -1) puts("No Solution");
    else printf("%lld\n",res);
  }
  return 0;
}