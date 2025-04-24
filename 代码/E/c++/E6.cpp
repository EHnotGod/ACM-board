#include<iostream>
#include<cstring>
using namespace std;

char a[200]="BCCABCCB";
char b[200]="AACCAB";
int f[201][201];

int main(){
  int ans=0;
  for(int i=1; i<=strlen(a); i++){
    for(int j=1; j<=strlen(b); j++){
      if(a[i-1]==b[j-1]) f[i][j]=f[i-1][j-1]+1;
      else f[i][j]=0;
      ans=max(ans,f[i][j]);
    }
  }
  printf("ans=%d\n",ans);
  return 0;
}