// 点从0~(n-1)，pa,pb记录最小割划分的两个部分的点


#include <stdio.h>
#include <string.h>
#include <vector>
#include <iostream>
#include <set>
#include <map>
#include <queue>
#define ll long long
#define clr(a,b) memset(a,b,sizeof(a))
using namespace std;

const int N = 505;

const int inf = 1e8;
int g[N][N];
bool vis[N],combine[N],par[N];
int d[N],node[N],st[N];
vector<int> vst[N];
vector<int> pa,pb;

int prim(int k){
    clr(vis,0);
    clr(d,0);
    int mincut = 0;
    int tmp = -1;
    int top = 0;
    for(int i=0;i<k;i++){
        int maxi = -inf;
        for(int j=0;j<k;j++){
            int u = node[j];
            if(!combine[u]&&!vis[u]&&d[u]>maxi){
                tmp = u;
                maxi = d[u];
            }
        }
        st[top++] = tmp;
        vis[tmp] = true;
        if(i==k-1)
            mincut=d[tmp];
        for(int j=0;j<k;j++){
            int u=node[j];
            if(!combine[u]&&!vis[u])
                d[u] += g[tmp][u];
        }
    }
    for(int i=0;i<top;i++)  node[i] = st[i];
    return mincut;
}
int Stoer_Wagner(int n){
    for(int i=0;i<n;i++){
        vst[i].clear();
        vst[i].push_back(i);
    }

    int ans = inf;
    clr(combine,0);
    for(int i=0;i<n;i++)
        node[i] = i;
    for(int i=1;i<n;i++){
        int tn = n-i+1;
        int cur = prim(tn);
        int s = st[tn-2], t= st[tn-1];
        if(cur<ans){
            ans =cur;
            for(int j=0;j<n;j++) par[j] = 0;
            for(int j=0;j<vst[t].size();j++){
                par[vst[t][j]] = 1;
            }

        }
        combine[t] = true;

        for(int j=0;j<vst[t].size();j++){
            vst[s].push_back(vst[t][j]);
        }
        for(int j=0;j<n;j++){
            if(j==s) continue;
            if(!combine[j]){
                g[s][j] += g[t][j];
                g[j][s] += g[j][t];
            }
        }
    }
    pa.clear();pb.clear();
    for(int i=0;i<n;i++)
        if(par[i]) pa.push_back(i);
        else pb.push_back(i);
    return ans;
}


int main(){
    int n,m;
    while(scanf("%d%d",&n,&m)!=EOF){
        clr(g,0);
        for(int i=0;i<m;i++){
            int u,v,w;
            scanf("%d%d%d",&u,&v,&w);
            g[u][v] += w;
            g[v][u] += w;
        }
        printf("%d\n",Stoer_Wagner(n));
    }
    return 0;
}

