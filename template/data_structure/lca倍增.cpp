//p为父亲节点，需要先build,查询a,距离dist(a,b)，ab最近公共祖先lca(a,b)

const int N = 1100000;
int f[N][20],n;
int d[N],p[N];

int lca(int a,int b){
    if(d[a]<d[b]) swap(a,b);
    int gap = d[a]-d[b];
    for(int i=0;i<20;i++){
        if((gap>>i)&1)
            a = f[a][i];
    }
    if(a==b) return a;
    int h=d[a];
    for(int i=19;i>=0;i--){
        if((h>>i)&1){
            if(f[a][i]==f[b][i])
                h = (1<<i)-1;
            else{
                a = f[a][i];
                b = f[b][i];
                h-=(1<<i);
            }
        }
    }
    return f[a][0];
}
int dist(int a,int b){
    int c = lca(a,b);
    return d[a]+d[b]-2*d[c];
}
void build(int n){
    for(int i=1;i<=n;i++) f[i][0]=p[i];
    for(int i=1;i<20;i++){
        for(int j=1;j<=n;j++){
            f[j][i] = f[f[j][i-1]][i-1];
        }
    }
}
