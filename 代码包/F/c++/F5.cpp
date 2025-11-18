#include <iostream>
#include <cstring>
#include <algorithm>
using namespace std;

const int N=3e7;
char a[N],s[N];
int d[N]; //回文半径函数

void get_d(char*s,int n){
  d[1]=1;
    for(int i=2,l,r=1;i<=n;i++){
        if(i<=r)d[i]=min(d[r-i+l],r-i+1);
        while(s[i-d[i]]==s[i+d[i]])d[i]++;
        if(i+d[i]-1>r)l=i-d[i]+1,r=i+d[i]-1;
        // printf("i=%d d=%d [%d %d]\n",i,d[i],l,r);
    }
}
int main(){
  //改造串
  scanf("%s",a+1);
  int n=strlen(a+1),k=0;
  s[0]='$',s[++k]='#';
  for(int i=1;i<=n;i++)
    s[++k]=a[i],s[++k]='#';
  n=k;

  get_d(s,n);//计算d函数
  int ans=0;
  for(int i=1;i<=n;i++)
    ans=max(ans,d[i]);
  printf("%d\n",ans-1);
  return 0;
}