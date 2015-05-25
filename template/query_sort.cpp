const int N = 100000;
int n,Q,ans;
struct item{
    int l,r,id;
    int nl,nr;
    ll ans;
}q[N];
int a[N];

bool cmp1(item a,item b){
    if(a.nl!=b.nl) return a.nl<b.nl;
    else if(a.nr!=b.nr){
        if(a.nl&1) return a.nr<b.nr;
        else return a.nr>b.nr;
    }
    else return max(a.l,a.r) < max(b.l,b.r);
}
bool cmp2(item a,item b){
    return a.id<b.id;
}
void add(int x){
}
void del(int x){
}

int main(){
//    freopen("/home/zyc/Documents/Code/cpp/in","r",stdin);
    int T,cas=0;
    scanf("%d",&T);
    while(T--){
        scanf("%d",&n);
        for(int i=1;i<=n;i++)
            scanf("%d",&a[i]);
        scanf("%d",&Q);
        int sqr = sqrt(1.0*n);
        for(int i=1;i<=Q;i++){
            scanf("%d%d",&q[i].l,&q[i].r);
            q[i].nl = q[i].l/sqr;
            q[i].nr = q[i].r/sqr;
            q[i].id = i;
            q[i].ans = 0;
        }
        sort(q+1,q+Q+1,cmp1);
        int l=0,r=0;
        ans = 0 ;
        for(int i=1;i<=Q;i++){
            while(l>q[i].l) add(--l);
            while(r<q[i].r) add(++r);
            while(l<q[i].l) del(l++);
            while(r>q[i].r) del(r--);
            q[i].ans = ans;
        }
        sort(q+1,q+Q+1,cmp2);

        printf("Case #%d:\n",++cas);
        for(int i=1;i<=Q;i++)
            printf("%lld\n",q[i].ans);
    }
    return 0;
}

