#include<cstdio>

int main(){
  int m; scanf("%d",&m);
  int i=1,j=1,sum=1;
  while(i<=m/2){      //i<=j
    if(sum<m){
      j++;
      sum+=j;
    }
    if(sum>=m){
      if(sum==m) printf("%d %d\n",i,j);
      sum-=i;
      i++;
    }
  }
  return 0;
}