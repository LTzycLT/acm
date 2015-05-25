const int N = 1000005 , M = 2000005;

int head[N],nxt[M],ev[M],ew[M],edge;
ll dis[N] ;
void add(int u,int v,ll w){
    ev[++edge]=v;ew[edge]=w;nxt[edge]=head[u];head[u]=edge;
    ev[++edge]=u;ew[edge]=w;nxt[edge]=head[v];head[v]=edge;
}
class Tree_Maxdis{
    public:
    ll dp[N][2];
    int from[N][2];

    void dfs1(int u,int f){
        dp[u][0] = dp[u][1] = 0;
        from[u][0] = from[u][1] = -1;
        for(int i=head[u];i!=-1;i=nxt[i]){
            int v = ev[i];
            ll w = ew[i];
            if(v==f) continue;
            dfs1(v,u);
            ll tw = dp[v][0] + w;
            if(tw > dp[u][0]){
                dp[u][1] = dp[u][0];
                from[u][1] = from[u][0];

                dp[u][0] = tw;
                from[u][0] = v;
            }else if(tw>dp[u][1]){
                dp[u][1] = tw;
                from[u][1] = v;
            }
        }
    }
    void dfs2(int u,int f,ll mx){
        dis[u] = max(dp[u][0],mx);
        for(int i=head[u];i!=-1;i=nxt[i]){
            int v = ev[i];
            ll w = ew[i];
            if(v==f) continue;
			int nw ;
            if(from[u][0] == v)
				nw = max(mx,dp[u][1])+w;
			else
				nw = max(mx,dp[u][0])+w;
			dfs2(v,u,nw);
        }
    }
    void solve(int s){
        dfs1(s,s);
        dfs2(s,s,0);
    }
}tree;

