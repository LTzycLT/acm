/*
index
0) template
1) 平衡树找第K大数，或数字为第几大
2) 子集遍历
3) 线段树 区间更新 区间求和
4) 主席树
5) mod int
6) ball box
7) shortest path
8) 二分匹配 匈牙利
9) Prime
10) pow2 逆元
11) 组合数
12) LCA 倍增
13) suffix array
14) 回文串
*/

/*==============================================================================*\
  1) template
  \*==============================================================================*/
mt19937 rng((unsigned int) chrono::steady_clock::now().time_since_epoch().count());
rand = rng() % n;

start = clock()
const double TL = 1.5 * CLOCKS_PER_SEC;
clock() - start < TL

/*==============================================================================*\
  1) 平衡树找第K大数，或数字为第几大
  \*==============================================================================*/

#include <iostream>
#include <ext/pb_ds/assoc_container.hpp>
#include <ext/pb_ds/tree_policy.hpp>
using namespace std;
using namespace __gnu_pbds;


//c++11 and more
//comparer: less less_equal greater greater_equal equal not_equal_to
template <typename T>
using oset = tree<T, null_type, less<T>, rb_tree_tag, tree_order_statistics_node_update>; //有序set
template <typename T>
using omultiset = tree<T, null_type, less_equal<T>, rb_tree_tag, tree_order_statistics_node_update>; //有序multiset
template <typename K, typename V>
using omap = tree<K, V, less<K>, rb_tree_tag, tree_order_statistics_node_update>; //有序map
//less than c++11
typedef tree<int, int, less<int>, rb_tree_tag, tree_order_statistics_node_update> map_t;

int main()
{
    oset<int> os;
    os.insert(10);
    os.insert(20);
    os.insert(30);
    os.insert(40);
    os.insert(50);
    //return the nth element
    if(os.find_by_order(3) != os.end()) {
        cout << *os.find_by_order(3) << endl; //output 40
    }
    cout << os.order_of_key(40) << endl; //output 3
    cout << os.order_of_key(44) << endl; //output 4

    omultiset<int> oms;
    oms.insert(1);
    oms.insert(1);
    oms.insert(2);
    oms.insert(4);
    oms.insert(4);
    cout << *oms.find_by_order(4) << endl; //out 3
    cout << oms.order_of_key(3) << endl; //out 3
    return 0;
}

/*==============================================================================*\
    2) 子集遍历
  \*==============================================================================*/

for(int l = (rt - 1) & rt; l > 0; l = (l - 1) & rt) {
    cout << l << " " << rt^l;
}

/*==============================================================================*\
    3) 线段树 区间更新 区间求和
  \*==============================================================================*/
#define LL long long
const int N = 200000;
LL val[N * 4], sum[N * 4], base[N];

void push_down(int rt, int sl, int m, int sr) {
    val[rt * 2] += val[rt];
    sum[rt * 2] += val[rt] * (m - sl + 1);
    val[rt * 2 + 1] += val[rt]; 
    sum[rt * 2 + 1] += val[rt] * (sr - m);
    val[rt] = 0;
}
void push_up(int rt) {
    sum[rt] = sum[rt * 2] + sum[rt * 2 + 1];
    sum[rt] = sum[rt * 2] + sum[rt * 2 + 1];
}
void build(int sl, int sr, int rt) {
  if(sl == sr) {
    val[rt] = 0;
    sum[rt] = base[sl];
    return ;
  }
  int m = (sl + sr) / 2;
  build(sl, m, rt * 2);
  build(m + 1, sr, rt * 2 + 1);
  push_up(rt);
}
void update(int l, int r, LL w, int sl, int sr, int rt) {
    //printf("%d %d %d %d\n", l, r, sl, sr);
    if(l == sl && r == sr) {
        val[rt] += w;
        sum[rt] += (r - l + 1) * w;
        return ;
    }
    int m = (sl + sr) / 2;
    push_down(rt, sl, m, sr);
    if(r <= m) update(l, r, w, sl, m, rt * 2);
    else if(l > m) update(l, r, w, m + 1, sr, rt * 2 + 1);
    else {
        update(l, m, w, sl, m, rt * 2);
        update(m + 1, r, w, m + 1, sr, rt * 2 + 1);
    }
    push_up(rt);
}

LL query(int l, int r, int sl, int sr, int rt) {
    if(l == sl && r == sr) {
        return sum[rt];
    }

    int m = (sl + sr) / 2;
    push_down(rt, sl, m, sr);
    LL ans;
    if(r <= m) ans = query(l, r, sl, m, rt * 2);
    else if(l > m) ans = query(l, r, m + 1, sr, rt * 2 + 1);
    else {
        ans = query(l, m, sl, m, rt * 2) + query(m + 1, r, m + 1, sr, rt * 2 + 1);
    }
    push_up(rt);
    return ans;
}

int main() {
    build(1, n, 1);
    update(l, r, w, 1, n, 1);
    query(l, r, 1, n, 1);
    return 0;
}

/*==============================================================================*\
    4) 主席树
  \*==============================================================================*/

int n, m;
map<int, int> mp, invmp;
int a[100005];
const int MAXN = 100005;
int rt[MAXN], v[MAXN << 6], ls[MAXN << 6], rs[MAXN << 6], tot;
void build(int l, int r, int &nd) {
    if (!nd) nd = ++tot;
    if (l == r) return;
    int m = (l + r) / 2;
    build(l, m, ls[nd]);
    build(m + 1, r, rs[nd]);
}
void update(int p, int x, int l, int r, int lnd, int &nd) {
    nd = ++tot, v[nd] = v[lnd] + x, ls[nd] = ls[lnd], rs[nd] = rs[lnd];
    if (l == r) return;
    int m = (l + r) / 2;
    if (p <= m)
        update(p, x, l, m, ls[lnd], ls[nd]);
    else
        update(p, x, m + 1, r, rs[lnd], rs[nd]);
}
//区间求和
int query(int ql, int qr, int l, int r, int nd) {
    if (ql == l && qr == r) return v[nd];
    int mid = (l + r) / 2;
    if (qr <= mid) return query(ql, qr, l, mid, ls[nd]);
    if (ql > mid) return query(ql, qr, mid + 1, r, rs[nd]);
    return query(ql, mid, l, mid, ls[nd]) + query(mid + 1, qr, mid + 1, r, rs[nd]);
}
//区间求第k大
int kth_element(int k, int l, int r, int lnd, int nd) {
    if(l == r) return l;
    int mid = (l + r) / 2;
    int lv = v[ls[nd]] - v[ls[lnd]];
    if(k <= lv) return kth_element(k, l, mid, ls[lnd], ls[nd]);
    else return kth_element(k - lv, mid + 1, r, rs[lnd], rs[nd]);
}

int main() {
    freopen("in", "r", stdin);
    scanf("%d%d", &n, &m);
  
    //数值重新hash
    for(int i = 1; i <= n; i++) {
        scanf("%d", &a[i]);
        mp[a[i]] = 0;
    }
    int cnt = 0;
    for(map<int, int>::iterator it = mp.begin(); it != mp.end(); it++) {
        invmp[cnt] = it->first;
        it->second = cnt++;
    }
    //建树
    build(0, cnt, rt[0]);
    //插入
    for(int i = 1; i <= n; i++) {
        update(mp[a[i]], 1, 0, cnt, rt[i - 1], rt[i]);
    }
    //区间第k大询问
    for(int i = 0; i < m; i++) {
        int l, r, k;
        scanf("%d%d%d", &l, &r, &k);
        int ans = kth_element(k, 0, cnt, rt[l - 1], rt[r]);
        printf("%d\n", invmp[ans]);
    }
    return 0;
}
 

/*==============================================================================*\
    5) mod int
  \*==============================================================================*/

const int MOD = 1e9 + 7;
struct mod_int {
	int val;

	mod_int(long long v = 0) {
		if(v < 0) v = v % MOD + MOD;
		if(v >= MOD) v %= MOD;
		val = v;
	}
	static int mod_inv(int a, int m = MOD) {
		int g = m, r = a, x = 0, y = 1;
		while(r != 0) {
			int q = g / r;
			g %= r; swap(g, r);
			x -= q * y; swap(x, y);
		}
		return x < 0 ? x + m: x;
	}
	explicit operator int() const {
		return val;
	}
	mod_int& operator+=(const mod_int &other) {
		val += other.val;
		if(val >= MOD) val -= MOD;
		return *this;
	}
	mod_int& operator-=(const mod_int &other) {
		val -= other.val;
		if (val < 0) val += MOD;
		return *this;
	}
	mod_int& operator*=(const mod_int &other) {
		val = (uint64_t) val * other.val % MOD;
		return *this;
	}
	mod_int& operator/=(const mod_int &other) {
		return *this *= other.inv();
	}
	friend mod_int operator+(const mod_int &a, const mod_int &b) {
		return mod_int(a) += b;
	}
	friend mod_int operator-(const mod_int &a, const mod_int &b) {
		return mod_int(a) -= b;
	}
	friend mod_int operator*(const mod_int &a, const mod_int &b) {
		return mod_int(a) *= b;
	}
	friend mod_int operator/(const mod_int &a, const mod_int &b) {
		return mod_int(a) /= b;
	}
	mod_int& operator++() {
		val = val == MOD - 1 ? 0: val + 1;
		return *this;
	}
	mod_int operator++(int) {
		mod_int before = *this;
		++*this;
		return before;
	}
	mod_int& operator--() {
		val = val == 0 ? MOD - 1 : val - 1;
		return *this;
	}
	mod_int operator--(int) {
		mod_int before = *this;
		--*this;
		return before;
	}
	mod_int operator--() const {
		return val == 0 ? 0 : MOD - val;
	}
	bool operator==(const mod_int &other) const {
		return val == other.val;
	}
	bool operator!=(const mod_int &other) const {
		return val != other.val;
	}
	mod_int inv() const {
		return mod_inv(val);
	}
	mod_int pow(long long p) const {
		//assert(p >= 0);
		mod_int a = *this, result = 1;
		while(p > 0) {
			if(p & 1) {
				result *= a;
			}
			a *= a;
			p >>= 1;
		}
		return result;
	}
}

/*==============================================================================*\
    6) ball box
  \*==============================================================================*/

mod_int factorial(int n) {
    mod_int result = 1;
    for(int i = 1; i <= n; i++) {
        result *= i;
    }
    return result.val;
}
mod_int combination(int n, int m) {
    for(int i = 0; i <= n; i++) {
        f[i][0] = f[i][1] = 1;
        for(int j = 1; j < m; j++) {
            f[i][j] = f[i - 1][j] + f[i - 1][j - 1];
        }
    }
    return f[n][m];
}

const int N = 1005;
mod_int f[N][N];
mod_int distinct_ball_same_box(int n, int m) {
    //not empty
    f[0][0] = 1;
    for(int i = 1; i <= n; i++) {
        for(int j = 1; j <= m; j++) {
            f[i][j] = f[i - 1][j - 1] + f[i - 1][j] * j;
        }
    }
    return f[n][m];

    //empty
    mod_int ans = 0;
    for(int i = 1; i <= m; i++) {
        ans += f[n][i];
    }
    return ans;
}

mod_int distinct_ball_distinct_box(int n, int m) {
    //not empty
    return factorial(m) * distinct_ball_same_box(n, m);

    //empty
    return mod_int(n).pow(m);
}

mod_int same_ball_distinct_box(int n, int m) {
    //not empty
    return combination(n - 1, m - 1);

    //empty
    return combination(n + m - 1, m - 1);
}

mod_int same_ball_same_box(int n, int m) {
    for(int i = 1; i <= n; i++) f[1][i] = 1;
    for(int i = 1; i <= m; i++) f[i][1] = 1;
    for (int i = 2; i <= n; i++) {
        for (int j = 2; j <= m; j++) {
            f[i][j] = f[i - 1][j];
            if(i >= j) {
                f[i][j] += f[i - j][j];
            }
        }
    }
    //empty
    return f[n][m];
    //not empty
    return f[n - m][m];
}

/*==============================================================================*\
    7) shortest path
  \*==============================================================================*/

const int inf = 1e9;

int n, m;
const int n = 1005, m = 100005;

int head[n], ev[m], ew[m], nxt[m], e; 

void addedge(int u, int v, int w) {
    ev[e] = v; ew[e] = w; nxt[e] = head[u]; head[u] = e++;
    ev[e] = u; ew[e] = w; nxt[e] = head[v]; head[v] = e++;
}

int d[n], vis[n];

int dij(int s, int t) {
    for(int i = 0; i < n; i++) d[i] = inf, vis[i] = false;;
    d[s] = 0;
    priority_queue<pair<int, int> > q;
    q.push(make_pair(0, s));
    while(q.size() > 0) {
        int u = q.top().second;
        q.pop();
        if(vis[u]) continue;
        vis[u] = true;
        for(int i = head[u]; i != -1; i = nxt[i]) {
            int v = ev[i], w = ew[i];
            if(d[u] + w < d[v]) {
                d[v] = d[u] + w;
                q.push(make_pair(-d[v], v));
            }
        }
    }
    return d[t] < inf ? d[t] : -1;
}

int spfa(int s, int t) {
    for(int i = 0; i < n; i++) d[i] = inf, vis[i] = false;;
    d[s] = 0;
    queue<int> q;
    q.push(s);
    vis[s] = true;
    while(q.size() > 0) {
        int u = q.front();
        vis[u] = false;
        q.pop();
        for(int i = head[u]; i != -1; i = nxt[i]) {
            int v = ev[i], w = ew[i];
            if(d[u] + w < d[v]) {
                d[v] = d[u] + w;
                if(!vis[v]) {
                    q.push(v);
                }
            }
        }
    }
    return d[t] < inf ? d[t] : -1;
}

/*==============================================================================*\
    8) 二分匹配 匈牙利
  \*==============================================================================*/

class Hungary {
    public:
    vector<int> match, used;
    vector<vector<int> > e;
    int ln, rn;
        
    void init(int _ln, int _rn) {
        ln = _ln;
        rn = _rn;
        match.clear(); used.clear(); e.clear();
        match = vector<int>(rn, -1);
        used = vector<int>(rn, 0);
        e = vector<vector<int> >(ln);
    }
    void addedge(int u, int v) {
        e[u].push_back(v);
    }
    bool dfs(int u) {
        for(auto v: e[u]) {
            if(!used[v]) {
                used[v] = 1; 
                if(match[v] == -1 || dfs(match[v])) {
                    match[v] = u; 
                    return true;
                }
            }
        }
        return false;
    }
    int run() {
        int ans = 0;
        for(int i = 0; i < rn; i++) match[i] = -1;
        for(int i = 0; i < ln; i++) {
            for(int j = 0; j < rn; j++) used[j] = 0;
            if(dfs(i)) ans += 1;
        }
        return ans;
    }
};
/*==============================================================================*\
    9) Prime

    得到质因子getPrimeFactor(x), 需要init(sqrt(x) + 1)
  \*==============================================================================*/


class Prime {
    public:
    vector<bool> f;
    vector<LL> p;
    void init(int maxx) {
        f = vector<bool>(maxx + 1, 0);
        p.clear();
        for(int i = 2; i <= maxx; i++) {
            if(f[i] == 0) {
                p.push_back(i);
                for(int j = i + i; j <= maxx; j += i) {
                    f[j] = 1;
                }
            }
        }
    }
    vector<LL> getPrimeFactor(LL x) {
        vector<LL> ans;
        for(auto prime: p) {
            if(prime * prime > x) break;
            if(x % prime == 0) {
                ans.push_back(prime);
                while(x % prime == 0) {
                    x /= prime;
                }
            }
        }
        if(x > 1) ans.push_back(x);
        return ans;
    }
};


/*==============================================================================*\
    10) pow2, 逆元
  \*==============================================================================*/
LL pow2(LL a, LL b) {
    LL ans = 1;
    while(b > 0) {
        if(b & 1) ans = (ans * a) % mod;
        a = (a * a) % mod;
        b >>= 1;
    }
    return ans;
}

LL inv(LL x) {
    return pow2(x, mod - 2);
}

/*==============================================================================*\
    11) 组合数 c[n][m]
  \*==============================================================================*/
class Combination{
    //mode=1: pre cal c[n][m]
    //mode=2: pre cal fac[] invfac[] => fac[n] * invfac[m] * invfac[n - m]
    public:
        Combination(int _mode, int _sz): mode(_mode), sz(_sz){
            if(mode == 1) {
                c = vector<vector<LL>>(sz + 1, vector<LL>(sz + 1, 0));
                c[0][0] = 1;
                for(int i = 1; i <= sz; i++) {
                    c[i][0] = c[i][i] = 1;
                    for(int j = 1; j < i; j++) {
                        c[i][j] = (c[i - 1][j] + c[i - 1][j - 1]) % mod;
                    }
                }
            }
            else {
                fac = vector<LL>(sz + 1);
                invfac = vector<LL>(sz + 1);
                fac[0] = invfac[0] = 1;
                for(int i = 1; i <= sz; i++) {
                    fac[i] = fac[i - 1] * i % mod;
                    invfac[i] = invfac[i - 1] * inv(i) % mod;
                }
            }
        }
        LL get(int n, int m){
            assert(m <= n && n <= sz);
            if(mode == 1) return c[n][m];
            else return fac[n] * invfac[m] % mod * invfac[n - m] % mod;
        }

        int mode, sz;
        vector<vector<LL>> c;
        vector<LL> fac, invfac;
};

/*==============================================================================*\
    12) LCA倍增
  \*==============================================================================*/

#define clr(x, y) memset(x, y, sizeof(x))
#define forn(i, n) for(int i = 0; i < n; i++)
 
const int N = 100005;
const int M = N * 2;
const int H = 20;
 
int e, head[N];
int ev[M], nxt[M];
struct graph {
    void init() {
        e = 0;
        clr(head, -1);
    }
    void addedge(int u, int v) {
        ev[e] = v;
        nxt[e] = head[u];
        head[u] = e++;
        ev[e] = u;
        nxt[e] = head[v];
        head[v] = e++;
    }
}g;

int ln[N];
int pnt[N][H], depth[N], stk[N];
struct LCA {
    void init() {  // 求1-N所有log2(x)的值，只需初始化一次
        ln[0] = ln[1] = 0;
        for (int i = 2; i < N; ++i)
            ln[i] = ln[i >> 1] + 1;
    }
    int getfather(int x, int len) {
        while (len > 0) {
            x = pnt[x][ln[len]];
            len -= 1 << ln[len];
        }
        return x;
    }
    int lca(int x, int y) {
        int low = 0, high = min(depth[x], depth[y]);
        x = getfather(x, depth[x] - high);
        y = getfather(y, depth[y] - high);
        if (x == y) return x;
            while (high - low > 1) {
            int mid = ln[high - low - 1];
            int nx = pnt[x][mid];
            int ny = pnt[y][mid];
            mid = high - (1 << mid);
            if (nx == ny)
                low = mid;
            else {
                high = mid;
                x = nx;
                y = ny;
            }
        }
        return pnt[x][ln[high - low]];
    }
    /********下面求得depth[]和pnt[][]值，也可以通过其他方式求得********/
     void build(const graph& g, int root, int n) {
        forn(i, n) {
            depth[i] = -1;
            clr(pnt[i], -1);
        }
        int top = 1;
        depth[stk[0] = root] = 0;
        while (top) {  // 这里默认g为一颗树，若为森林需要修改此处
            int u = stk[--top];
            for (int i = head[u]; ~i; i = nxt[i]) {
                int v = ev[i];
                if (depth[v] != -1) continue;
                stk[top++] = v;
                pnt[v][0] = u;
                depth[v] = depth[u] + 1;
            }
        }
        for (int i = 1; i < H; ++i)
            forn(u, n) if (pnt[u][i - 1] != -1)
                pnt[u][i] = pnt[pnt[u][i - 1]][i - 1];
    }
}lca;
 
int n, m;
 
int dis(int u, int v) {
    int f = lca.lca(u, v);
    return depth[u] + depth[v] - 2 * depth[f];
}
int main() {
    //freopen("in", "r", stdin);
    g.init();
    g.addedge(u, v);
    lca.init();
    lca.build(g, 0, n);
    lca.lca(u, v);
    return 0;
}

/*==============================================================================*\
    13) suffix array
  \*==============================================================================*/

/*
   srank[0...7]: 4 6 8 1 2 3 5 7
string:
a a b a a a a b
------------------------------------------sa[1] = 3 : a a a a b
height[1] = 0
sa[2] = 4 : a a a b
height[2] = 3
sa[3] = 5 : a a b
height[3] = 2
sa[4] = 0 : a a b a a a a b height[4] = 3
sa[5] = 6 : a b
height[5] = 1
sa[6] = 1 : a b a a a a b
height[6] = 2
sa[7] = 7 : b
height[7] = 0
sa[8] = 2 : b a a a a b
height[8] = 1
*/
const int N = 200010;
int ua[N], ub[N], us[N], sa[N];
int cmp(int *r,int a,int b,int l){
    return r[a]==r[b]&&r[a+l]==r[b+l];
}
void da(int *r,int *sa,int n,int m){ //da(r, sa, n + 1, 256);(r[n] = 0)
    int i,j,p,*x=ua,*y=ub,*t;
    //r[]存放原字符串，且从char变为int
    for(i=0;i<m;i++) us[i]=0; //sa[i]表示排名为i的后缀起始下标(i>=1,sa[i]>=0)
    for(i=0;i<n;i++) us[x[i]=r[i]]++;
    for(i=1;i<m;i++) us[i]+=us[i-1];

    for(i=n-1;i>=0;i--) sa[--us[x[i]]]=i;
    for(j=1,p=1;p<n;j*=2,m=p){
        for(p=0,i=n-j;i<n;i++) y[p++]=i;
        for(i=0;i<n;i++) if(sa[i]>=j) y[p++]=sa[i]-j;
        for(i=0;i<m;i++) us[i]=0;
        for(i=0;i<n;i++) us[x[i]]++;
        for(i=1;i<m;i++) us[i]+=us[i-1];
        for(i=n-1;i>=0;i--) sa[--us[x[y[i]]]]=y[i];
        for(t=x,x=y,y=t,p=1,x[sa[0]]=0,i=1;i<n;i++)
            x[sa[i]]=cmp(y,sa[i-1],sa[i],j)?p-1:p++;
    }
}
int srank[N],height[N]; //height[i]为排第i-1和第i的后缀的公共前缀长度
void calheight(int *r,int *sa,int n){
    int i,j,k=0;
    for(i=1;i<=n;i++) srank[sa[i]]=i;
    for(i=0;i<n;height[srank[i++]]=k)
        for(k?k--:0,j=sa[srank[i]-1];r[i+k]==r[j+k];k++);
}
int *RMQ = height; // RMQ为查询的数组,这里RMQ=height
//int RMQ[N];
int mm[N];
int best[20][N]; //best[i][j]表示[j, j + 2^i)区间中的最小值
void initRMQ(int n){
    int i,j,a,b;
    for(mm[0]=-1,i=1;i<=n;i++)
        mm[i]=((i&(i-1))==0)?mm[i-1]+1:mm[i-1];
    for(i=1;i<=n;i++) best[0][i]=i;
    for(i=1;i<=mm[n];i++)
        for(j=1;j<=n+1-(1<<i);j++){
            a=best[i-1][j];
            b=best[i-1][j+(1<<(i-1))];
            if(RMQ[a]<RMQ[b]) best[i][j]=a;
            else best[i][j]=b;
        }
}
int askRMQ(int a,int b){
    int t;
    t=mm[b-a+1];b-=(1<<t)-1;
    a=best[t][a];b=best[t][b];
    return RMQ[a]<RMQ[b]?a:b;
}
int lcp(int a,int b, int n){ //后缀r[a]和r[b]的公共前缀长度
    int t;
    if(a == b) return n - a;
    a=srank[a];b=srank[b];
    if(a>b) {t=a;a=b;b=t;}
    return(height[askRMQ(a+1,b)]);
}

int main() {
  int n;
  //s can't contain 0 values
  s[n] = 0;
  da(s, sa, n + 1, 256);
  calheight(s, sa, n);
  initRMQ(n);
  lcp(l, r, n);
}

/*==============================================================================*\
    14) 回文串
  \*==============================================================================*/

/*
   原串： w a a b w s w f d
   新串r[]： $ # w # a # a # b # w # s # w # f # d #
   辅助数组P： 1 2 1 2 3 2 1 2 1 2 1 4 1 2 1 2 1 2 1
   p[id]- 1 就是该回文子串在原串中的长度
   */

const int N = 1100100 * 2;
int r[N], p[N];
void pk(int *r, int n, int *p) {
    int i, id, mx = 0;
    for (i = 1; i < n; ++i) {
        if (mx > i) p[i] = min(p[2 * id - i], mx - i);
        else p[i] = 1;

        for (; r[i + p[i]] == r[i - p[i]]; p[i]++);
        if (p[i] + i > mx) {
            mx = p[i] + i;
            id = i;
        }
    }
}
string solve(string str) {
    int len = str.size();
    int n = 0;
    r[n++] = '$'; r[n++] = '#';
    forn (i, len) {
        r[n++] = str[i];
        r[n++] = '#';
    }
    r[n] = 0;
    pk(r, n, p);

    //每一个长度>=2的回文串在原串中的位置
    for(int i = 2; i < n - 1; i++) {
      if(p[i] >= 3) {
        int l;
        if(i % 2 == 0) l = (i / 2 - 1) - (p[i] - 2) / 2;
        else l = ((i + 1) / 2 - 1) - (p[i] - 1) / 2;
        int r = l + p[i] - 2;
      }
    }
}

