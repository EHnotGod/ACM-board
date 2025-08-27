#include <iostream>
#include <cstring>
#include <algorithm>
using namespace std;

const int N=1010;
int n,a[N],b[N];

int main(){
  int t; cin>>t;
  while(t--){
    cin>>n;
    for(int i=0; i<n; i++) cin>>a[i]>>b[i];
    sort(a,a+n); sort(b,b+n);

    int x=a[n/2]-a[(n-1)/2]+1;
    int y=b[n/2]-b[(n-1)/2]+1;
    cout<<1LL*x*y<<'\n';
  }
  return 0;
}