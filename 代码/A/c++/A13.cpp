#include <iostream>
using namespace std;

int n,k,a[5000010];

int qnth_element(int l, int r){
  if(l==r) return a[l];
  int i=l-1, j=r+1, x=a[(l+r)/2];
  while(i<j){
    do i++; while(a[i]<x); //向右找>=x的数
    do j--; while(a[j]>x); //向左找<=x的数
    if(i<j) swap(a[i],a[j]);
  }
  if(k<=j) return qnth_element(l,j);
  else return qnth_element(j+1,r);
}

int main(){
  scanf("%d%d",&n,&k);
  for(int i=0;i<n;i++) scanf("%d",&a[i]);
  printf("%d\n",qnth_element(0,n-1));
}