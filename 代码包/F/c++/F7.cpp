// 01Trie 最大异或对
#include <iostream>
using namespace std;

const int N=100010;
int n, a[N];
int ch[N*31][2],cnt;

void insert(int x){
  int p=0;
  for(int i=30; i>=0; i--){
    int j=x>>i&1; //取出第i位
    if(!ch[p][j])ch[p][j]=++cnt;
    p=ch[p][j];
  }
}
int query(int x){
  int p=0,res=0;
  for(int i=30; i>=0; i--){
    int j=x>>i&1;
    if(ch[p][!j]){
      res += 1<<i; //累加位权
      p=ch[p][!j];
    }
    else p=ch[p][j];
  }
  return res;
}
int main(){
  cin>>n;
  for(int i=1; i<=n; i++)
    cin>>a[i],insert(a[i]);
  int ans=0;
  for(int i=1; i<=n; i++)
    ans=max(ans,query(a[i]));
  cout<<ans;
  return 0;
}