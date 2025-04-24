//分数规划+二分+排序 复杂度：nlogn*log(1e4)
#include <iostream>
#include <cstring>
#include <algorithm>
using namespace std;

const int N=1010;
int n,k;
double a[N], b[N], c[N];

bool check(double x){
  double s=0;
  for(int i=1;i<=n;++i)c[i]=a[i]-x*b[i];
  sort(c+1, c+n+1);
  for(int i=k+1;i<=n;++i) s+=c[i];
  return s>=0;
}
double find(){
  double l=0, r=1;
  while(r-l>1e-4){
    double mid=(l+r)/2;
    if(check(mid)) l=mid;//最大化
    else r=mid;
  }
  return l;
}
int main(){
  while(scanf("%d%d",&n,&k),n){
    for(int i=1;i<=n;i++)scanf("%lf",&a[i]);
    for(int i=1;i<=n;i++)scanf("%lf",&b[i]);
    printf("%.0lf\n", 100*find());
  }
  return 0;
}