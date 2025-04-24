// 我喜欢的板子
#include<cstdio>
using namespace std;
int n,m,q,a[1000005];

int find(int q){
    int l=0,r=n+1;//开区间
    while(l+1<r){ //l+1=r时结束
        int mid=l+r>>1;
        if(a[mid]>=q) r=mid; //最小化
        else l=mid;
    }
    return a[r]==q ? r : -1;
}
int main(){
    scanf("%d %d",&n,&m);
    for(int i=1;i<=n;i++)scanf("%d",&a[i]);
    for(int i=1;i<=m;i++)
        scanf("%d",&q), printf("%d ",find(q));
    return 0;
}