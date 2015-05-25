const int inf = 1e8;
const int N;
bool vis[N];
int w[N],sw[N];
vector<int> e[N],point;

void dfs(int u,int f){
    point.push_back(u);
    w[u] = sw[u] = 0;
    for(int i=0;i<e[u].size();i++){
        int v = e[u][i];
        if(vis[v]||v==f) continue;
        dfs(v,u);
        w[u] = max(sw[u],sw[v]);
        sw[u] += sw[v];
    }
    sw[u]++;
}
int find(int u){
    point.clear();
    dfs(u,u);
    int ma=inf,ans = -1;
    for(int i=0;i<point.size();i++){
        int v = point[i];
        Max(w[v],point.size()-sw[v]);
        if(w[v]<ma){
            ma = w[v];
            ans = v;
        }
    }
    return ans;
}


void dac(int u){
    int x = find(u);
    vis[x] = 1;
    for(int i=0;i<e[x].size();i++){
        int v = e[x][i];
        if(vis[v]) continue;
        dac(v);
    }
}
int main(){
    dac(0);
    return 0;
}

