#include <iostream>
#include <deque>
using namespace std;

const int N=1000010;
int a[N];
deque<int> q;

int main(){
  int n, k; scanf("%d%d", &n, &k);
  for(int i=1; i<=n; i++) scanf("%d", &a[i]);

  // 维护窗口最小值
  q.clear();                              //清空队列
  for(int i=1; i<=n; i++){                //枚举序列
    while(!q.empty() && a[q.back()]>=a[i]) q.pop_back(); //队尾出队(队列不空且新元素更优)
    q.push_back(i);                       //队尾入队(存储下标 方便判断队头出队)
    while(q.front()<i-k+1) q.pop_front(); //队头出队(队头元素滑出窗口)
    if(i>=k) printf("%d ",a[q.front()]);  //使用最值
  }
  puts("");

  // 维护窗口最大值
  q.clear();
  for(int i=1; i<=n; i++){
    while(!q.empty() && a[q.back()]<=a[i]) q.pop_back();
    q.push_back(i);
    while(q.front()<i-k+1) q.pop_front();
    if(i>=k) printf("%d ",a[q.front()]);
  }
}