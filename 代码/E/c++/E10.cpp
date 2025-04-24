// 二进制分组优化
#include<iostream>
using namespace std;

const int N=100005;
int n,m,a,b,s;
int v[N],w[N];
int f[N];

int main(){
  cin>>n>>m;

  int cnt=0;
  for(int i=1;i<=n;i++){
    cin>>b>>a>>s;
    for(int j=1;j<=s;j<<=1){
      v[++cnt]=j*a; w[cnt]=j*b;
      s-=j;
    }
    if(s) v[++cnt]=s*a, w[cnt]=s*b;
  }

  for(int i=1;i<=cnt;i++)
    for(int j=m;j>=v[i];j--)
      f[j]=max(f[j],f[j-v[i]]+w[i]);
  cout<<f[m];
}