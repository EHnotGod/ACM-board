#include <iostream>
#include <cstring>
#include <algorithm>
using namespace std;

const int N=4001;
int a[N],b[N],c[N];
int la,lb,lc;

void mul(int a[],int b[],int c[]){ //a*b=c
    for(int i=1;i<=la;i++)
        for(int j=1;j<=lb;j++)
            c[i+j-1]+=a[i]*b[j]; //存乘积

    for(int i=1;i<lc;i++){
        c[i+1]+=c[i]/10;  //存进位
        c[i]%=10;         //存余数
    }
    while(c[lc]==0&&lc>1) lc--; //去0
}
int main(){
    char A[N],B[N];
    cin>>A>>B;
    la=strlen(A); lb=strlen(B); lc=la+lb;
    for(int i=1;i<=la;i++)a[i]=A[la-i]-'0';
    for(int i=1;i<=lb;i++)b[i]=B[lb-i]-'0';
    mul(a,b,c);
    for(int i=lc;i>=1;i--) cout<<c[i];
    return 0;
}