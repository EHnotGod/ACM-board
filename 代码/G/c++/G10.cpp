#include <iostream>
using namespace std;

const int N = 1000010;
int p[N], vis[N], cnt;
int a[N]; //a[i]记录i的最小质因子的次数
int d[N]; //d[i]记录i的约数个数

void get_d(int n){ //筛法求约数个数
  d[1] = 1;
  for(int i=2; i<=n; i++){
    if(!vis[i]){
      p[++cnt] = i;
      a[i] = 1;
      d[i] = 2;
    }
    for(int j=1; i*p[j]<=n; j++){
      int m = i*p[j];
      vis[m] = 1;
      if(i%p[j] == 0){
        a[m] = a[i]+1;
        d[m] = d[i]/a[m]*(a[m]+1);
        break;
      }
      else{
        a[m] = 1;
        d[m] = d[i]*2;
      }
    }
  }
}
int main(){
  int n;
  cin >> n;
  get_d(n);
  for(int i=1; i<=n; i++)
    printf("%d\n", d[i]);
  return 0;
}