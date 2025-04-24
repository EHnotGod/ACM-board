#include <cstring>
#include <iostream>
#include <algorithm>
using namespace std;

const int N=1000010;
int m,n;
char S[N],P[N];
int nxt[N];

int main(){
  cin>>S+1>>P+1;
  m=strlen(S+1),n=strlen(P+1);

  nxt[1]=0;
  for(int i=2,j=0;i<=n;i++){
    while(j && P[i]!=P[j+1]) j=nxt[j];
    if(P[i]==P[j+1]) j++;
    nxt[i]=j;
  }

  for(int i=1,j=0;i<=m;i++){
    while(j && S[i]!=P[j+1]) j=nxt[j];
    if(S[i]==P[j+1]) j++;
    if(j==n) printf("%d\n",i-n+1);
  }

  for(int i=1;i<=n;i++)printf("%d ",nxt[i]);
  return 0;
}