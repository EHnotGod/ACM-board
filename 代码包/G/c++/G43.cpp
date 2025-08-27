// 迭代版 1.5s
#include <iostream>
#include <cstring>
#include <algorithm>
#include <cmath>
#define LL long long
using namespace std;
const int N=4e6;
const int g=3,P=998244353;
int n,m,R[N],gi,ni;
LL A[N],B[N];

LL qpow(LL a,LL b){
  LL ans=1;
  for(;b; a=a*a%P,b>>=1)
    if(b&1) ans=ans*a%P;
  return ans;
}
void NTT(LL A[],int n,int op){
  for(int i=0; i<n; ++i)
    R[i]=R[i/2]/2+((i&1)?n/2:0);
  for(int i=0; i<n; ++i)
    if(i<R[i]) swap(A[i],A[R[i]]);
  for(int i=2; i<=n; i<<=1){
    LL g1=qpow(op==1?g:gi,(P-1)/i);
    for(int j=0; j<n; j+=i){
      LL gk=1;
      for(int k=j; k<j+i/2; ++k){
        LL x=A[k], y=gk*A[k+i/2]%P;
        A[k]=(x+y)%P;A[k+i/2]=(x-y+P)%P;
        gk=gk*g1%P;
      }
    }
  }
}
int main(){
  scanf("%d%d",&n,&m);
  for(int i=0;i<=n;++i)scanf("%lld",A+i);
  for(int i=0;i<=m;++i)scanf("%lld",B+i);
  for(m=n+m,n=1; n<=m; n<<=1);
  gi=qpow(g,P-2); ni=qpow(n,P-2);
  NTT(A,n,1); NTT(B,n,1);
  for(int i=0;i<n;++i)A[i]=A[i]*B[i]%P;
  NTT(A,n,-1);
  for(int i=0;i<=m;++i)
    printf("%d ",A[i]*ni%P);
}