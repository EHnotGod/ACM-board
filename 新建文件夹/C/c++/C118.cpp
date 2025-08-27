// 李超线段树 O(nlogn)
#include <iostream>
#include <cstring>
#include <algorithm>
using namespace std;

#define N 50005
#define ls u<<1
#define rs u<<1|1
int n,cnt;
struct line{
  double k,b; //斜率,截距
}p[N*2];
int tr[N*4]; //线段编号

double Y(int id,int x){ //求Y值
  return p[id].k*x+p[id].b;
}
void change(int u,int l,int r,int id){ //修改  int mid=(l+r)>>1;
  if(Y(id,mid)>Y(tr[u],mid)) swap(id,tr[u]);
  if(Y(id,l)>Y(tr[u],l)) change(ls,l,mid,id);
  if(Y(id,r)>Y(tr[u],r)) change(rs,mid+1,r,id);
}
double query(int u,int l,int r,int x){ //查询
  if(l==r) return Y(tr[u],x);
  int mid=(l+r)>>1;
  double t=Y(tr[u],x);
  if(x<=mid) return max(t,query(ls,l,mid,x));
  else return max(t,query(rs,mid+1,r,x));
}
int main(){
  scanf("%d",&n);
  for(int i=1;i<=n;i++){
    char op[10]; scanf("%s",op);
    if(op[0]=='P'){
      double b,k; scanf("%lf%lf",&b,&k);
      p[++cnt]={k,b-k};
      change(1,1,N,cnt);
    }
    else{
      int x; scanf("%d",&x);
      printf("%d\n",(int)query(1,1,N,x)/100);
    }
  }
}