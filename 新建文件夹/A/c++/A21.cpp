#include<iostream>
#include<cstdio>
#include<algorithm>
using namespace std;

#define N 20005
struct line{    //线段
  int l,r;
  bool operator<(line &t){
    return l<t.l;
  }
}a[N];
int n,st,ed,sum;
//a[] 存储每条线段的起点,终点
//st  存储合并区间的起点
//ed  存储合并区间的终点
//sum 存储合并区间的长度

int main(){
  scanf("%d",&n);
  for(int i=1;i<=n;i++)
    cin>>a[i].l>>a[i].r;
  sort(a+1,a+n+1); //按起点排序

  st=a[1].l; ed=a[1].r;
  sum+=a[1].r-a[1].l;
  for(int i=2; i<=n; i++){
    if(a[i].l<=ed){
      if(a[i].r<ed) //覆盖
        continue;
      else {        //重叠
        st=ed;
        ed=a[i].r;
        sum+=ed-st;
      }
    }
    else{           //相离
      st=a[i].l;
      ed=a[i].r;
      sum+=ed-st;
    }
  }
  cout<<sum<<endl;
  return 0;
}