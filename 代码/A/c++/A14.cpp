#include <iostream>
using namespace std;

int n,a[500010],b[500010];
long long res;

void merge(int l,int r){
  if(l>=r) return;
  int mid=l+r>>1;
  merge(l,mid);
  merge(mid+1,r); //拆分

  int i=l,j=mid+1,k=l; //合并
  while(i<=mid && j<=r){
    if(a[i]<=a[j]) b[k++]=a[i++];
    else b[k++]=a[j++], res+=mid-i+1;
  }
  while(i<=mid) b[k++]=a[i++];
  while(j<=r) b[k++]=a[j++];
  for(i=l; i<=r; i++) a[i]=b[i];
}

int main(){
  cin>>n;
  for(int i=0;i<n;i++) scanf("%d",&a[i]);
  merge(0,n-1);
  printf("%lld\n",res);
}