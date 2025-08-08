#include <iostream>
#include <algorithm>
#include <cmath>
using namespace std;

const int N=110;
const double eps=1e-6;
int n;
double a[N][N]; //增广矩阵

int gauss(){
  int c,r;//当前列，当前行
  for(c=r=0; c<n; c++){
    //1.找到c列的最大行t
    int t=r;
    for(int i=r; i<n; i++)
      if(fabs(a[i][c])>fabs(a[t][c])) t=i;
    if(fabs(a[t][c])<eps) continue; //c列已0化

    //2.把最大行换到上面
    for(int i=c; i<n+1; i++)swap(a[t][i],a[r][i]);

    //3.把当前行r的第一个数，变成1
    for(int i=n; i>=c; i--)a[r][i]/=a[r][c];

    //4.把当前列c下面的所有数，全部消成0
    for(int i=r+1; i<n; i++)
      if(fabs(a[i][c])>eps)
        for(int j=n; j>=c; j--)
          a[i][j]-=a[i][c]*a[r][j];
    r++; //这一行的工作做完，换下一行
  }
  if(r<n){ //说明已经提前变成梯形矩阵
    for(int i=r; i<n; i++)
      if(fabs(a[i][n])>eps) //a[i][n]==bi
        return 2; //左边=0，右边≠0,无解
    return 1; //0==0，无穷多解
  }
  //5.唯一解，从下往上回代，得到方程的解
  for(int i=n-1; i>=0; i--)
    for(int j=i+1; j<n; j++)
      a[i][n]-=a[i][j]*a[j][n];
  return 0;
}
int main(){
  cin >> n;
  for(int i=0; i<n; i++)
    for(int j=0; j<=n; j++)
      cin >> a[i][j];
  int t=gauss();
  if(t) puts("No Solution");
  else for(int i=0; i<n; i++)
        printf("%.2lf\n",a[i][n]);
}