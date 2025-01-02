#include <bits/stdc++.h>
using namespace std;

#define rep(i,a,b) for(int i=a;i<b;++i)
#define vii vector<double>
#define lr 0.01
#define epochs 1000

double f(double x) {return 1/(1+exp(-x));}
double df(double x) {return x*(1-x);}
double randW() {return ((double)rand())/RAND_MAX;}

int main() {
    vii x={1,2,3,4,5},y={2,4,6,8,10};int n=x.size();
    double w=randW(),b=randW();

    rep(e,0,epochs){
        double L=0,gw=0,gb=0;
        rep(i,0,n){
            double p=w*x[i]+b,e=p-y[i];
            L+=e*e,gw+=e*x[i],gb+=e;
        }
        w-=lr*gw/n,b-=lr*gb/n;
    }

    cout<<"y = "<<w<<"x + "<<b<<endl;
    rep(i,0,n) cout<<x[i]<<": "<<y[i]<<" -> "<<w*x[i]+b<<endl;
    return 0;
}
