#include <iostream>
using namespace std;

const int N = 1000010;
int p[N], vis[N], cnt;
int phi[N];

void get_phi(int n){//筛法求欧拉函数
  phi[1] = 1;
  for(int i=2; i<=n; i++){
    if(!vis[i]){
      p[cnt++] = i;
      phi[i] = i-1;
    }
    for(int j=0; i*p[j]<=n; j++){
      int m = i*p[j];
      vis[m] = 1;
      if(i%p[j] == 0){
        phi[m] = p[j]*phi[i];
        break;
      }
      else
        phi[m]=(p[j]-1)*phi[i];
    }
  }
}
int main(){
  int n;
  cin >> n;
  get_phi(n);
  for(int i=1; i<=n; i++)
    printf("%d\n", phi[i]);
  return 0;
}