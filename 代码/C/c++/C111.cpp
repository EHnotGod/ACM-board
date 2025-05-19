// 普通莫队 O(n*sqrt(n))
#include <iostream>
#include <cstring>
#include <algorithm>
#include <cmath>
using namespace std;

const int N=50005;
int n,m,k,B,a[N];
int sum,c[N],ans[N];
struct Q{
  int l,r,id;
  //按l所在块的编号l/B和r排序
  bool operator<(Q &b){
    if(l/B!=b.l/B) return l<b.l;
    if((l/B)&1) return r<b.r;
    return r>b.r;
  }
}q[N];

void add(int x){ //扩展一个数
  sum-=c[x]*c[x];
  c[x]++;
  sum+=c[x]*c[x];
}
void del(int x){ //删除一个数
  sum-=c[x]*c[x];
  c[x]--;
  sum+=c[x]*c[x];
}
int main(){
  scanf("%d%d%d",&n,&m,&k);
  B=sqrt(n); //块的大小
  for(int i=1;i<=n;++i)scanf("%d",&a[i]);
  for(int i=1;i<=m;++i)
    scanf("%d%d",&q[i].l,&q[i].r),q[i].id=i;
  sort(q+1,q+1+m); //按l/B和r排序
  for(int i=1,l=1,r=0;i<=m;++i){
    while(l>q[i].l) add(a[--l]);//左扩展
    while(r<q[i].r) add(a[++r]);//右扩展
    while(l<q[i].l) del(a[l++]);//左删除
    while(r>q[i].r) del(a[r--]);//右删除
    ans[q[i].id]=sum;
  }
  for(int i=1;i<=m;++i)printf("%d\n",ans[i]);
}