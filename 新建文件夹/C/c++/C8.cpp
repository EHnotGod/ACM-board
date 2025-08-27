// 主席树 O(nlognlogn)
#include <iostream>
#include <cstring>
#include <algorithm>
using namespace std;

#define N 200005
#define lc(x) tr[x].l
#define rc(x) tr[x].r
struct node{
  int l,r,s; //s:节点值域中有多少个数
}tr[N*20];
int root[N],idx;
int n,m,a[N],b[N];

void insert(int x,int &y,int l,int r,int pos){
  y=++idx; //开点
  tr[y]=tr[x]; tr[y].s++;
  if(l==r) return;
  int m=l+r>>1;
  if(pos<=m) insert(lc(x),lc(y),l,m,pos);
  else insert(rc(x),rc(y),m+1,r,pos);
}
int query(int x,int y,int l,int r,int k){
  if(l==r) return l;
  int m=l+r>>1;
  int s=tr[lc(y)].s-tr[lc(x)].s;
  if(k<=s) return query(lc(x),lc(y),l,m,k);
  else return query(rc(x),rc(y),m+1,r,k-s);
}
int main(){
  scanf("%d%d",&n,&m);
  for(int i=1; i<=n; i++){
    scanf("%d",&a[i]); b[i]=a[i];
  }
  sort(b+1,b+n+1);
  int bn=unique(b+1,b+n+1)-b-1; //去重后的个数

  for(int i=1; i<=n; i++){
    int id=lower_bound(b+1,b+bn+1,a[i])-b;//下标
    insert(root[i-1],root[i],1,bn,id);
  }
  while(m--){
    int l,r,k; scanf("%d%d%d",&l,&r,&k);
    int id=query(root[l-1],root[r],1,bn,k);
    printf("%d\n",b[id]);
  }
}