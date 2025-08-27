#include<bits/stdc++.h>
using namespace std;

const int N=114514,inf=INT_MAX;
const double eps=1e-9;

int n;
int a[N],b[N],c[N];
inline double f(double x){
	double maxn=-inf;
	for(int i=1;i<=n;i++){
		maxn=max(maxn,a[i]*x*x+b[i]*x+c[i]);
	}
	return maxn;
}

signed main() {
	int t;
	cin>>t;
	while(t--){
		cin>>n;
		for(int i=1;i<=n;i++){
			cin>>a[i]>>b[i]>>c[i];
		}

		double l=0,r=1000;
		while(r-l>eps){
			double m1=(2*l+r)/3;
			double m2=(l+2*r)/3;
			if(f(m1)<f(m2)){
				r=m2;
			}else{
				l=m1;
			}
		}

		cout<<fixed<<setprecision(4)<<f(l)<<endl;
	}
	return 0;
}