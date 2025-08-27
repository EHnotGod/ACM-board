#include <iostream>
#include <cstring>
#include <algorithm>
using namespace std;

const int N=30;
int n, ans;
int pos[N],c[N],p[N],q[N];

void print(){
  if(ans<=3){
    for(int i=1;i<=n;i++)
      printf("%d ",pos[i]);
    puts("");
  }
}
void dfs(int i){
  if(i>n){
    ans++; print(); return;
  }
  for(int j=1; j<=n; j++){
    if(c[j]||p[i+j]||q[i-j+n])continue;
    pos[i]=j; //记录第i行放在了第j列
    c[j]=p[i+j]=q[i-j+n]=1; //宣布占领
    dfs(i+1);
    c[j]=p[i+j]=q[i-j+n]=0; //恢复现场
  }
}
int main(){
  cin >> n;
  dfs(1);
  cout << ans;
  return 0;
}