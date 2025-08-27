//二维费用 01背包
#include <iostream>
using namespace std;

int f[110][110];
// f[j,k]:前i个物品，体积≤j，重量≤k 的最大价值

int main(){
  int n, V, W;    //物品 容量 承重
  int v, w, val;  //体积 重量 价值
  cin>>n>>V>>W;
  for(int i=1; i<=n; i++){  //物品
    cin>>v>>w>>val;
    for(int j=V; j>=v; j--) //体积
    for(int k=W; k>=w; k--) //重量
      f[j][k]=max(f[j][k],f[j-v][k-w]+val);
  }
  cout<<f[V][W];
}