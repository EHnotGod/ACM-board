#include <iostream>
#include <cstring>
using namespace std;

typedef long long LL;
const int N=100010;
int a[N], b[N];

int main(){
  int n;
  cin>>n;
  for(int i=1; i<=n; i++) cin>>a[i];
  for(int i=1; i<=n; i++) b[i]=a[i]-a[i-1];

  LL p=0, q=0;
  for(int i=2; i<=n; i++)
    if(b[i]>0) p+=b[i];
    else q+=abs(b[i]);

  cout<<max(p,q)<<'\n'<<abs(p-q)+1;
}