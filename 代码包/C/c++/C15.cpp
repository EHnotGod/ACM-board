// 扫描线+线段树+离散化 1.4s
#include <iostream>
#include <cstdio>
#include <algorithm>
using namespace std;

#define ls u<<1
#define rs u<<1|1
const int N=200005;
struct line{   //扫描线
  int x1,x2,y;
  int tag;     //入边:+1,出边:-1
  bool operator<(line &t){return y<t.y;}
}L[N];
int cnt[N*8],len[N*8]; //线段树
int X[N];              //X坐标
void pushup(int u,int l, int r){
  if(cnt[u]) len[u]=X[r+1]-X[l]; //r → X[r+1]
  else len[u]=len[ls]+len[rs];
}
void change(int u,int l,int r,int a,int b,int tag){
  if(a>r || b<l) return; //越界
  if(a<=l && r<=b){      //覆盖
    cnt[u]+=tag;
    pushup(u,l,r);
    return;
  }
  int m=l+r>>1;
  change(ls,l,m,a,b,tag); //裂开
  change(rs,m+1,r,a,b,tag);
  pushup(u,l,r);
}
int main(){
  int n,x1,x2,y1,y2; scanf("%d",&n);
  for(int i=1; i<=n; i++){
    scanf("%d%d%d%d",&x1,&y1,&x2,&y2);
    L[i]={x1,x2,y1,1};
    L[n+i]={x1,x2,y2,-1};
    X[i]=x1; X[n+i]=x2;
  }
  n*=2;
  sort(L+1,L+n+1); //扫描线排序
  sort(X+1,X+n+1); //X坐标排序
  int s=unique(X+1,X+n+1)-X-1; //去重

  long long ans=0;
  for(int i=1; i<n; i++){
    int l=lower_bound(X+1,X+s+1,L[i].x1)-X;
    int r=lower_bound(X+1,X+s+1,L[i].x2)-X;
    change(1,1,s,l,r-1,L[i].tag); //x2 → r-1
    ans+=1ll*(L[i+1].y-L[i].y)*len[1];
  }
  printf("%lld\n",ans);
}