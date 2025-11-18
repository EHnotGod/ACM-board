// 单调队列+DP O(n)
#include<bits/stdc++.h>
using namespace std;

const int N=200010;
int n,m,a[N];
int q[N],f[N];

int main(){
  cin>>n>>m;
  for(int i=1; i<=n; i++) cin>>a[i];

  int ans=2e9;
  for(int i=1,h=1,t=0; i<=n; i++){
    while(h<=t && q[h]<i-m) h++;
    while(h<=t && f[q[t]]>=f[i-1]) t--;
    q[++t]=i-1;
    f[i]=f[q[h]]+a[i];
    if(i>=n-m+1) ans=min(ans,f[i]);
  }
  cout<<ans;
}