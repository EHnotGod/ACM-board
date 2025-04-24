#include <iostream>
#include <cstring>
#include <algorithm>
using namespace std;

struct line{
  int l,r; //线段的左,右端点
  bool operator<(line &b){
    return r<b.r;
  }
}L[1000005];
int n,last,cnt;

int main(){
  scanf("%d",&n);
  for(int i=1;i<=n;i++)
    scanf("%d%d",&L[i].l,&L[i].r);

  sort(L+1,L+n+1); //右端点排序
  for(int i=1;i<=n;i++){
    if(last<=L[i].l){
      last=L[i].r;
      cnt++;
    }
  }
  printf("%d\n",cnt);
  return 0;
}