#include <iostream>
#include <cstring>
#include <algorithm>
using namespace std;

int n,m;
int a[100005],s[100005];

int main(){
    scanf("%d",&n);
    for(int i=1;i<=n;i++){
      scanf("%d",&a[i]);
        s[i]=s[i-1]+a[i];
    }

    scanf("%d",&m);
    for(int i=1;i<=m;i++){
        int l,r; scanf("%d%d",&l,&r);
        printf("%d\n",s[r]-s[l-1]);
    }
}