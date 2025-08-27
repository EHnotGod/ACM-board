#include<bits/stdc++.h>
using namespace std;

const int N=210,M=10010;
int n,m,a,b;
int h[N],to[M],ne[M],idx=1; //从2,3开始配对
void add(int a,int b){
  to[++idx]=b;ne[idx]=h[a];h[a]=idx;
}
int dfn[N],low[N],tim,cnt;
struct bridge{
  int x,y;
  bool operator<(const bridge &t)const{
    if(x==t.x) return y<t.y;
    return x<t.x;
  }
}bri[M]; //割边

void tarjan(int x,int ine){
  dfn[x]=low[x]=++tim;
  for(int i=h[x];i;i=ne[i]){
    int y=to[i];
    if(!dfn[y]){ //若y未访问
      tarjan(y,i);
      low[x]=min(low[x],low[y]);
      if(low[y]>dfn[x]) bri[cnt++]={x,y};
    }
    else if(i!=(ine^1)) //若y已访问且不是反边
      low[x]=min(low[x],dfn[y]);
  }
}
int main(){
  cin>>n>>m;
  while(m--){
    cin>>a>>b;
    add(a,b),add(b,a);
  }
  for(int i=1;i<=n;i++) if(!dfn[i])tarjan(i,0);
  sort(bri,bri+cnt);
  for(int i=0;i<cnt;i++)
    cout<<bri[i].x<<" "<<bri[i].y<<"\n";
}