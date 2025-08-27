#include <iostream>
#include <cstring>
#include <algorithm>
using namespace std;

const int N=12;
int a[N];     //把整数的每一位数字抠出来，存入数组
int f[N][N];  //f[i][j]表示一共有i位，且最高位数字是j的不降数的个数

void init(){  //预处理不降数的个数
  for(int i=0; i<=9; i++) f[1][i]=1;  //一位数
  for(int i=2; i<N; i++)        //阶段：枚举位数
    for(int j=0; j<=9; j++)     //状态：枚举最高位
      for(int k=j; k<=9; k++)   //决策：枚举次高位
        f[i][j]+=f[i-1][k];
}
int dp(int n){
  if(!n) return 1;              //特判，n==0返回1
  int cnt=0;
  while(n) a[++cnt]=n%10, n/=10;//把每一位抠出来存入数组a

  int res=0, last=0;            //last表示上一位数字
  for(int i=cnt; i>=1; --i){    //从高位到低位枚举
    int now=a[i];               //now表示当前位数字
    for(int j=last; j<now; j++) //枚举当前位可填入的数字
      res += f[i][j];           //累加答案
    if(now<last) break;         //若小，则break
    last=now;                   //更新last
    if(i==1) res++;             //特判，走到a1的情况
  }
  return res;
}
int main(){
  init();     //预处理不降数的个数
  int l,r;
  while(cin>>l>>r) cout<<dp(r)-dp(l-1)<<endl;
  return 0;
}