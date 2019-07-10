/*

1) 平衡树找第K大数，或数字为第几大
2) 子集遍历

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
    3) 主席树
  \*==============================================================================*/

int n, m;
map<int, int> mp, invmp;
int a[100005];
const int MAXN = 100005;
int rt[MAXN], v[MAXN << 6], ls[MAXN << 6], rs[MAXN << 6], tot;
void build(int l, int r, int &n) {
    if (!n) n = ++tot;
    if (l == r) return;
    int m = (l + r) / 2;
    build(l, m, ls[n]);
    build(m + 1, r, rs[n]);
}
void update(int p, int x, int l, int r, int ln, int &n) {
    n = ++tot, v[n] = v[ln] + x, ls[n] = ls[ln], rs[n] = rs[ln];
    if (l == r) return;
    int m = (l + r) / 2;
    if (p <= m)
        update(p, x, l, m, ls[ln], ls[n]);
    else
        update(p, x, m + 1, r, rs[ln], rs[n]);
}
//区间求和
int query(int ql, int qr, int l, int r, int n) {
    if (ql == l && qr == r) return v[n];
    int m = (l + r) / 2;
    if (qr <= m) return query(ql, qr, l, m, ls[n]);
    if (ql > m) return query(ql, qr, m + 1, r, rs[n]);
    return query(ql, m, l, m, ls[n]) + query(m + 1, qr, m + 1, r, rs[n]);
}
//区间求第k大
int kth_element(int k, int l, int r, int ln, int n) {
    if(l == r) return l;
    int m = (l + r) / 2;
    int lv = v[ls[n]] - v[ls[ln]];
    if(k <= lv) return kth_element(k, l, m, ls[ln], ls[n]);
    else return kth_element(k - lv, m + 1, r, rs[ln], rs[n]);
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
