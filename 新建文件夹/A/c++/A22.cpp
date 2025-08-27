#include <cstdio>
#include <queue>
using namespace std;

const int N=100005;
int a[N],b[N],id[N];
priority_queue<pair<int,int>,
       vector<pair<int,int>>,
       greater<pair<int,int>>>q;
//id[i]: 记录b[i]的搭档的下标
//q: 小根堆，存储<两数和,组的下标>

int main(){
  int n; scanf("%d",&n);
  for(int i=1; i<=n; i++) scanf("%d",&a[i]);
  for(int i=1; i<=n; i++){
    scanf("%d",&b[i]);
    id[i]=1;
    q.push({a[1]+b[i],i});
  }
  while(n--){
    printf("%d ",q.top().first);
    int i=q.top().second; q.pop();
    q.push({a[++id[i]]+b[i],i});
  }
  return 0;
}