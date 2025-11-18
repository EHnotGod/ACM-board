#include <cstdio>
#include <cstring>
#include <set>
using namespace std;

const int N=2005,M=10005;
int n,m,k,a,b,x;
int h[N],to[M],ne[M],tot; //邻接表
int f[N];

void add(int a,int b){
  to[++tot]=b,ne[tot]=h[a],h[a]=tot;
}
int sg(int u){
  // 记忆化搜索
  if(f[u]!=-1) return f[u];
  // 把子节点的sg值插入集合
  set<int> S;
  for(int i=h[u];i;i=ne[i])
    S.insert(sg(to[i]));
  // mex运算求当前节点的sg值并记忆
  for(int i=0; ;i++)
    if(!S.count(i)) return f[u]=i;
}
int main(){
  scanf("%d%d%d",&n,&m,&k);
  for(int i=0;i<m;i++)
    scanf("%d%d",&a,&b), add(a,b);
  memset(f,-1,sizeof f);
  int res=0;
  for(int i=0;i<k;i++)
    scanf("%d",&x),res^=sg(x);
  if(res) puts("win");
  else puts("lose");
}