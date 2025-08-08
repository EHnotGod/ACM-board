#include <iostream>
#include <cstring>
#include <algorithm>
using namespace std;

typedef long long LL;
const int mod=1000000007;
struct matrix{
    LL c[101][101];
    matrix(){memset(c, 0, sizeof c);}
} A, res;
LL n, k;

matrix operator*(matrix &x, matrix &y){ //矩阵乘法
    matrix t; //临时矩阵
    for(int i=1; i<=n; i++)
      for(int j=1; j<=n; j++)
        for(int k=1; k<=n; k++)
          t.c[i][j]=(t.c[i][j]+x.c[i][k]*y.c[k][j])%mod;
    return t;
}
void quickpow(LL k){ //快速幂
    for(int i=1; i<=n; i++) res.c[i][i]=1; //单位矩阵
    while(k){
        if(k & 1) res = res*A;
        A = A*A;
        k >>= 1;
    }
}
int main(){
    scanf("%d%lld",&n,&k);
    for(int i=1; i<=n; i++)
        for(int j=1; j<=n; j++)
            scanf("%d",&A.c[i][j]);
    quickpow(k);
    for(int i=1; i<=n; i++){
        for(int j=1; j<=n; j++)
            printf("%d ",res.c[i][j]);
        puts("");
    }
    return 0;
}