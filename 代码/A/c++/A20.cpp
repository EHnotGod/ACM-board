#include <iostream>
#include <cstring>
#include <algorithm>
using namespace std;

typedef long long LL;
LL a[200005];
LL s1,s2,ans;
//s1:算术和, s2:异或和, ans:方案数

int main(){
  int n; cin>>n;
  for(int i=1; i<=n; i++) cin>>a[i];

  for(int i=1,j=0; i<=n; ){  //i<=j
    while(j+1<=n&&s1+a[j+1]==(s2^a[j+1])){ //预判
      j++;
      s1+=a[j];
      s2^=a[j];
    }
    ans+=j-i+1;
    s1-=a[i];
    s2^=a[i];
    i++;
  }
  cout<<ans<<endl;
  return 0;
}