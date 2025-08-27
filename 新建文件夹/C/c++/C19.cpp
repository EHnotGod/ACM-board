// 交替建树 970ms
#include <cstdio>
#include <cstring>
#include <algorithm>
#include <cmath>
#define lc t[p].l
#define rc t[p].r
using namespace std;

const int N=200010;
double ans=2e18;
int n,K,root,cur; //K维度,root根,cur当前节点
struct KD{     //KD树节点信息
  int l,r;     //左右孩子
  double v[2]; //点的坐标值
  double L[2],U[2]; //子树区域的坐标范围
  bool operator<(const KD &b)const{return v[K]<b.v[K];}
}t[N];

void pushup(int p){ //更新p子树区域的坐标范围
  for(int i=0;i<2;i++){
    t[p].L[i]=t[p].U[i]=t[p].v[i];
    if(lc)
      t[p].L[i]=min(t[p].L[i],t[lc].L[i]),
      t[p].U[i]=max(t[p].U[i],t[lc].U[i]);
    if(rc)
      t[p].L[i]=min(t[p].L[i],t[rc].L[i]),
      t[p].U[i]=max(t[p].U[i],t[rc].U[i]);
  }
}
int build(int l,int r,int k){ //交替建树
  if(l>r) return 0;
  int m=(l+r)>>1;
  K=k; nth_element(t+l,t+m,t+r+1); //中位数
  t[m].l=build(l,m-1,k^1);
  t[m].r=build(m+1,r,k^1);
  pushup(m);
  return m;
}
double sq(double x){return x*x;}
double dis(int p){ //当前点到p点的距离
  double s=0;
  for(int i=0;i<2;i++)
    s+=sq(t[cur].v[i]-t[p].v[i]);
  return s;
}
double dis2(int p){ //当前点到p子树区域的最小距离
  if(!p) return 2e18;
  double s=0;
  for(int i=0;i<2;++i)
    s+=sq(max(t[cur].v[i]-t[p].U[i],0.0))+
       sq(max(t[p].L[i]-t[cur].v[i],0.0));
  return s;
}
void query(int p){ //查询当前点的最小距离
  if(!p) return;
  if(p!=cur) ans=min(ans,dis(p));
  double dl=dis2(lc),dr=dis2(rc);
  if(dl<dr){
    if(dl<ans) query(lc);
    if(dr<ans) query(rc);
  }
  else{
    if(dr<ans) query(rc);
    if(dl<ans) query(lc);
  }
}
int main(){
  scanf("%d",&n);
  for(int i=1; i<=n; i++)
    scanf("%lf%lf",&t[i].v[0],&t[i].v[1]);
  root=build(1,n,0);
  for(cur=1; cur<=n; cur++) query(root);
  printf("%.4lf\n",sqrt(ans));
}