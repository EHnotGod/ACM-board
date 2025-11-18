#include <iostream>
#include <cstring>
#include <algorithm>
#include <cmath>
using namespace std;

const double eps=1e-10;
double a,b,c,d,l,r;

double f(double x){ //积分函数
  return (c*x+d)/(a*x+b);
}
double simpson(double l,double r){//辛普森公式
  return (r-l)*(f(l)+f(r)+4*f((l+r)/2))/6;
}
double asr(double l,double r,double ans){//自适应
  auto m=(l+r)/2,a=simpson(l,m),b=simpson(m,r);
  if(fabs(a+b-ans)<eps) return ans;
  return asr(l,m,a)+asr(m,r,b);
}
int main(){
  scanf("%lf%lf%lf%lf%lf%lf",&a,&b,&c,&d,&l,&r);
  printf("%.6lf",asr(l,r,simpson(l,r)));
  return 0;
}