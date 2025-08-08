// 迭代版 1.5s
#include <iostream>
#include <cstring>
#include <algorithm>
#include <cmath>
using namespace std;
const int N=4e6;
const double PI=acos(-1);

struct complex{
  double x, y;
  complex operator+(const complex& t)const{
    return {x+t.x, y+t.y};}
  complex operator-(const complex& t)const{
    return {x-t.x, y-t.y};}
  complex operator*(const complex& t)const{
    return {x*t.x-y*t.y, x*t.y+y*t.x};}
}A[N], B[N];
int R[N];

void FFT(complex A[],int n,int op){
  for(int i=0; i<n; ++i)
    R[i] = R[i/2]/2 + ((i&1)?n/2:0);
  for(int i=0; i<n; ++i)
    if(i<R[i]) swap(A[i],A[R[i]]);
  for(int i=2; i<=n; i<<=1){
    complex w1({cos(2*PI/i),sin(2*PI/i)*op});
    for(int j=0; j<n; j+=i){
      complex wk({1,0});
      for(int k=j; k<j+i/2; ++k){
        complex x=A[k], y=A[k+i/2]*wk;
        A[k]=x+y; A[k+i/2]=x-y;
        wk=wk*w1;
      }
    }
  }
}
int main(){
  int n,m;
  scanf("%d%d", &n, &m);
  for(int i=0; i<=n; i++)scanf("%lf", &A[i].x);
  for(int i=0; i<=m; i++)scanf("%lf", &B[i].x);
  for(m=n+m,n=1;n<=m;n<<=1);
  FFT(A,n,1); FFT(B,n,1);
  for(int i=0;i<n;++i)A[i]=A[i]*B[i];
  FFT(A,n,-1);
  for(int i=0;i<=m;++i)
    printf("%d ",(int)(A[i].x/n+0.5));
}