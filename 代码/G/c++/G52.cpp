#include <iostream>
#include <cstring>
#include <algorithm>
#include <cmath>
using namespace std;

const int N=100010;
struct Point{double x,y;} p[N],s[N];
int n,top;

double cross(Point a,Point b,Point c){ //叉积
  return (b.x-a.x)*(c.y-a.y)-(b.y-a.y)*(c.x-a.x);
}
double dis(Point a,Point b){ //距离
  return sqrt((a.x-b.x)*(a.x-b.x)+(a.y-b.y)*(a.y-b.y));
}
bool cmp(Point a, Point b){ //比较
  return a.x!=b.x ? a.x<b.x : a.y<b.y;
}
double Andrew(){
  sort(p+1,p+n+1,cmp); //排序
  for(int i=1; i<=n; i++){ //下凸包
    while(top>1&&cross(s[top-1],s[top],p[i])<=0)top--;
    s[++top]=p[i];
  }
  int t=top;
  for(int i=n-1; i>=1; i--){ //上凸包
    while(top>t&&cross(s[top-1],s[top],p[i])<=0)top--;
    s[++top]=p[i];
  }
  double res=0; //周长
  for(int i=1; i<top; i++) res+=dis(s[i],s[i+1]);
  return res;
}
int main(){
  scanf("%d",&n);
  for(int i=1;i<=n;i++)scanf("%lf%lf",&p[i].x,&p[i].y);
  printf("%.2lf\n", Andrew());
  return 0;
}