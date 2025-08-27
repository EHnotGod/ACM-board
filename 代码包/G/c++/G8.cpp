#include <iostream>
#include <cstring>
#include <algorithm>
using namespace std;

const int N = 100000010;
int vis[N];        // 划掉合数
int prim[N];       // 记录素数
int spf[N];        // 新增：记录每个数的最小质因数
int cnt;           // 素数个数

void get_prim(int n){ // 线性筛法
  for(int i = 2; i <= n; i++){
    if(!vis[i]){
      prim[++cnt] = i;
      spf[i] = i;    // 素数的最小质因数就是自身
    }
    for(int j = 1; 1LL * i * prim[j] <= n; j++){
      int m = i * prim[j];
      vis[m] = 1;
      spf[m] = prim[j];  // 记录 m 的最小质因数
      if(i % prim[j] == 0) break;
    }
  }
}

int main(){
    int n, q, k;
    scanf("%d %d", &n, &q);
    get_prim(n);
    while(q--){
        scanf("%d", &k);
        printf("%d\n", prim[k]);
    }
    return 0;
}
