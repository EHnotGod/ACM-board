#include <iostream>
#include <cstring>
#include <algorithm>
using namespace std;

const int N=50005;
struct node{
  int w,s;
  bool operator<(node &t){
    return w+s<t.w+t.s;
  }
}a[N];

int main(){
  int n; cin>>n;
  for(int i=1; i<=n; i++)
    cin>>a[i].w>>a[i].s;
  sort(a+1,a+n+1);

  int res=-2e9, t=0;
  for(int i=1; i<=n; i++){
    res=max(res,t-a[i].s);
    t+=a[i].w;
  }
  cout<<res<<endl;
}