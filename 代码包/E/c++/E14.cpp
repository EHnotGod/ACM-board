#include <iostream>
using namespace std;

int f[1010];

int main(){
  int n, m, v, w, s;
  scanf("%d %d", &n, &m);
  for(int i=1; i<=n; i++){  //枚举物品种类
    scanf("%d%d%d",&v,&w,&s);
    if(s!=0){               //01背包或多重背包
      if(s==-1) s=1;
      int num=min(s,m/v);
      for(int k=1; num>0; k<<=1){
        if(k>num) k=num;
        num-=k;
        for(int j=m; j>=v*k; j--)
          f[j]=max(f[j],f[j-v*k]+w*k);
      }
    }
    else{                   //完全背包
      for(int j=v; j<=m; j++)
        f[j]=max(f[j],f[j-v]+w);
    }
  }
  printf("%d\n", f[m]);
}