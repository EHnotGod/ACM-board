#include <iostream>
#include <cstring>
#include <algorithm>
using namespace std;

const int N=100010;
int n, a[N];
int len, b[N]; //记录上升子序列

int main(){
  scanf("%d", &n);
  for(int i=0; i<n; i++) scanf("%d", &a[i]);

  b[0]=-2e9;                              //哨兵
  for(int i=0; i<n; i++)
    if(b[len]<a[i]) b[++len]=a[i];        //新数大于队尾数，则插入队尾
    else *lower_bound(b,b+len,a[i])=a[i]; //替换第一个大于等于a[i]的数(贪心)

  printf("%d\n", len);
}