#include <iostream>
#include <cstring>
#include <algorithm>
using namespace std;

typedef long long LL;
const int N = 500010;
int n,m,q[N];
LL s[N],f[N];

LL dy(int i,int j){return f[i]+s[i]*s[i]-f[j]-s[j]*s[j];}
LL dx(int i,int j){return s[i]-s[j];}
int main(){
  while(~scanf("%d%d",&n,&m)){
    for(int i=1;i<=n;i++)scanf("%lld",&s[i]),s[i]+=s[i-1];

    int h=1,t=0;
    for(int i=1;i<=n;i++){
      while(h<t && dy(i-1,q[t])*dx(q[t],q[t-1])
                 <=dx(i-1,q[t])*dy(q[t],q[t-1])) t--;
      q[++t]=i-1;
      while(h<t && dy(q[h+1],q[h])
                 <=dx(q[h+1],q[h])*2*s[i]) h++;
      int j=q[h];
      f[i]=f[j]+(s[i]-s[j])*(s[i]-s[j])+m;
    }
    printf("%lld\n",f[n]);
  }
}