#include <iostream>
#include <cstring>
#include <algorithm>
using namespace std;

const int N=5005;
int a[N],b,c[N];
int len;

void div(int a[],int b,int c[]){ //a/b=c
    long long t=0;
    for(int i=len;i>=1;i--){
        t=t*10+a[i];  //被除数
        c[i]=t/b;     //存商
        t%=b;         //余数
    }
    while(c[len]==0&&len>1) len--; //去0
}
int main(){
    char A[N]; cin>>A>>b; len=strlen(A);
    for(int i=1;i<=len;i++) a[i]=A[len-i]-'0';
    div(a,b,c);
    for(int i=len;i;i--) cout<<c[i];
    return 0;
}