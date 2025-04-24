#include <iostream>
using namespace std;

const int N=20000;
int a[N],b[N],c[N];
int la,lb,lc;

bool cmp(int a[],int b[]){
    if(la!=lb) return la<lb;
    for(int i=la; i; i--)
        if(a[i]!=b[i]) return a[i]<b[i];
    return false;  //相等时,避免-0
}
void sub(int a[],int b[],int c[]){ //a-b=c
    for(int i=1; i<=lc; i++){
        if(a[i]<b[i]) a[i+1]--,a[i]+=10;
        c[i]=a[i]-b[i];
    }
    while(c[lc]==0&&lc>1) lc--; //去0
}
int main(){
    string sa,sb; cin>>sa>>sb;
    la=sa.size(),lb=sb.size(),lc=max(la,lb);
    for(int i=1;i<=la;i++) a[i]=sa[la-i]-'0';
    for(int i=1;i<=lb;i++) b[i]=sb[lb-i]-'0';
    if(cmp(a,b)) swap(a,b),cout<<'-';
    sub(a,b,c);
    for(int i=lc;i;i--) printf("%d",c[i]);
    return 0;
}