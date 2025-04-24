#include<cstring>
#include<iostream>
#include<algorithm>
#include<queue>
using namespace std;

int main(){
  int n,w; scanf("%d%d",&n,&w); //选手总数,获奖率
  priority_queue<int> a; //大根堆
  priority_queue<int,vector<int>,greater<int> > b;

  for(int i=1; i<=n; i++){
    int x; scanf("%d",&x);
    if(b.empty()||x>=b.top()) b.push(x); //插入
    else a.push(x);

    int k=max(1,i*w/100); //第k大
    while(b.size()>k) a.push(b.top()), b.pop(); //调整
    while(b.size()<k) b.push(a.top()), a.pop();
    printf("%d ", b.top()); //取值
  }
}