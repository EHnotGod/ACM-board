#include <iostream>
using namespace std;

const int N = 1000010;
int p[N], vis[N], cnt;
int mu[N];

void get_mu(int n){//筛法求莫比乌斯函数
  mu[1] = 1;
  for(int i=2; i<=n; i++){
    if(!vis[i]){
      p[++cnt] = i;
      mu[i] = -1;
    }
    for(int j=1; i*p[j]<=n; j++){
      int m = i*p[j];
      vis[m] = 1;
      if(i%p[j] == 0){
        mu[m] = 0;
        break;
      }
      else
        mu[m] = -mu[i];
    }
  }
}
int main(){
  int n;
  cin >> n;
  get_mu(n);
  for(int i=1; i<=n; i++)
    printf("%d\n",mu[i]);
  return 0;
}