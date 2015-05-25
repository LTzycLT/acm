const int N = 200000,M=500000,Q=500000;
class ext_tarjan{
    public:
    bool vis[N];
    int head[N],nxt[M],ev[M],ew[M],edge;
    int pd[N],fa[N];
    int ans[Q],lu[Q],lv[Q];
    vector<pair<int,int> > q[N];
    vector<int> fq[N];


    void init(int n){
        edge = -1;
        for(int i=1;i<=n;i++){
            head[i] = -1;
            pd[i] = 0;
            fa[i] = i;
            vis[i] = 0;
            q[i].clear();
            fq[i].clear();
        }
    }
    void addedge(int u,int v,int w){
        ev[++edge]=v; ew[edge]=w; nxt[edge]=head[u]; head[u]=edge;
		ev[++edge]=u; ew[edge]=w; nxt[edge]=head[v]; head[v]=edge;
    }
    int find(int u){
        if(fa[u]==u) return u;
        int p=fa[u];
        fa[u]=find(fa[u]);
        pd[u]=max(pd[u],pd[p]);
        return fa[u];
    }
    void addquery(int u,int v,int id){
        lu[id] = u;
        lv[id] = v;
        q[u].push_back(make_pair(v,id));
        q[v].push_back(make_pair(u,id));
    }
    void tarjan(int u,int f){
        vis[u]=true;
        for(int i=head[u];i!=-1;i=nxt[i]){
            int v=ev[i];
            if(v==f) continue;
            tarjan(v,u);
            pd[v]=ew[i];
            fa[v]=u;
        }
        for(int i=0;i<(int)q[u].size();i++){
            int v=q[u][i].first,id=q[u][i].second;
            if(vis[v]){
                int p=find(v);
                fq[p].push_back(id);
            }
        }
        for(int i=0;i<(int)fq[u].size();i++){
            int id = fq[u][i];
            int x=lu[id],y=lv[id];
            find(x);find(y);
            ans[id]=max(pd[x],pd[y]);
        }
    }
}tj;
