#include <iostream>
#include <cstring>
#include <algorithm>
using namespace std;

typedef long long LL;
const int N = 100005;
int n, k, a[N];

bool check(int x){
  LL y=0; //段数
  for(int i=1;i<=n;i++) y+=a[i]/x;
  return y>=k; //x小,y大
}
int find(){
  int l=0, r=1e8+1;
  while(l+1<r){
    int mid=l+r>>1;
    if(check(mid)) l=mid; //最大化
    else r=mid;
  }
  return l;
}
int main(){
  scanf("%d%d",&n,&k);
  for(int i=1; i<=n; i++)scanf("%d",&a[i]);
  printf("%d\n",find());
  return 0;
}