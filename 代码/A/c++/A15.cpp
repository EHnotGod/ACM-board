// STL代码
#include <iostream>
#include <cstring>
#include <algorithm>
#include <queue>
using namespace std;

priority_queue<int,vector<int>,greater<int> > q;

int main(){
  int n; scanf("%d",&n); //操作次数
  while(n--){
    int op,x; scanf("%d",&op);
    if(op==1) scanf("%d",&x), q.push(x);
    else if(op==2) printf("%d\n",q.top());
    else q.pop();
  }
}