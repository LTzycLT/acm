/*
index
1) 平衡树找第K大数，或数字为第几大
2) 子集遍历
3) 线段树 区间更新 区间求和
4) 主席树
5) mod int
6) ball box
7) shortest path
8) 二分匹配 匈牙利
*/

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

LL val[N], sum[N];
void update(int l, int r, LL w, int sl, int sr, int rt) {
    //printf("%d %d %d %d\n", l, r, sl, sr);
    if(l == sl && r == sr) {
        val[rt] += w;
        sum[rt] += (r - l + 1) * w;
        return ;
    }
    int m = (sl + sr) / 2;
    if(r <= m) update(l, r, w, sl, m, rt * 2);
    else if(l > m) update(l, r, w, m + 1, sr, rt * 2 + 1);
    else {
        update(l, m, w, sl, m, rt * 2);
        update(m + 1, r, w, m + 1, sr, rt * 2 + 1);
    }
    sum[rt] = (sr - sl + 1) * val[rt] + sum[rt * 2] + sum[rt * 2 + 1];
}

LL query(int l, int r, int sl, int sr, int rt) {
    if(l == sl && r == sr) {
        return sum[rt];
    }

    int m = (sl + sr) / 2;

    val[rt * 2] += val[rt];
    sum[rt * 2] += val[rt] * (m - sl + 1);
    val[rt * 2 + 1] += val[rt]; 
    sum[rt * 2 + 1] += val[rt] * (sr - m);
    val[rt] = 0;

    LL ans;
    if(r <= m) ans = query(l, r, sl, m, rt * 2);
    else if(l > m) ans = query(l, r, m + 1, sr, rt * 2 + 1);
    else {
        ans = query(l, m, sl, m, rt * 2) + query(m + 1, r, m + 1, sr, rt * 2 + 1);
    }
    sum[rt] = (sr - sl + 1) * val[rt] + sum[rt * 2] + sum[rt * 2 + 1];
    return ans;
}

int main() {
    for(int i = 1; i <= 4 * n; i++) {
        val[i] = sum[i] = 0;
    }

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
