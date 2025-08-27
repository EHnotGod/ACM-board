#include<iostream>
#include<cstring>
using namespace std;

const int N=100010, M=110;
int w[N], f[N][M][2];

int main(){
  int n, k; cin >> n >> k;
  for(int i=1; i<=n; i++) cin >> w[i];

  for(int j=0; j<=k; j++) f[0][j][1]=-1e6;

  for(int i=1; i<=n; i++)
  for(int j=1; j<=k; j++){
    f[i][j][0]=max(f[i-1][j][0], f[i-1][j][1]+w[i]);
    f[i][j][1]=max(f[i-1][j][1], f[i-1][j-1][0]-w[i]);
  }

  cout << f[n][k][0];
}