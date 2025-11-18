// 树状数组 区修+点查 O(nlogn)
#include<bits/stdc++.h>
using namespace std;

const int N=500010;
int n,m,a[N];

struct BIT{
  int s[N]; //差分的区间和
  void change(int x,int w){
    for(;x<=n;x+=x&-x) s[x]+=w;
  }
  int query(int x){
    int res=0;
    for(;x;x-=x&-x) res+=s[x];
    return res;
  }
}T;
int main(){
  ios::sync_with_stdio(0);
  cin>>n>>m; int op,x,y,k;
  for(int i=1;i<=n;i++) cin>>a[i];
  for(int i=1;i<=m;i++){
    cin>>op>>x;
    if(op==1){
      cin>>y>>k;
      T.change(x,k);
      T.change(y+1,-k); //差分
    }
    else cout<<T.query(x)+a[x]<<"\n";
  }
}