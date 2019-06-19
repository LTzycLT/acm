#include <iostream>
#include <ext/pb_ds/assoc_container.hpp>
#include <ext/pb_ds/tree_policy.hpp>
using namespace std;
using namespace __gnu_pbds;

//c++11 and more
//comparer: less less_equal greater greater_equal equal not_equal_to
template <typename T>
using oset = tree<T, null_type, less<T>, rb_tree_tag, tree_order_statistics_node_update>;

template <typename K, typename V>
using omap = tree<K, V, less<K>, rb_tree_tag, tree_order_statistics_node_update>;

//less than c++11
typedef tree<int, int, less<int>, rb_tree_tag, tree_order_statistics_node_update> map_t;

int main()
{
    oset<int> active;
    active.insert(10);
    active.insert(20);
    active.insert(30);
    active.insert(40);
    active.insert(50);
    //return the nth element
    if(active.find_by_order(4) != active.end()) {
        cout << *active.find_by_order(4) << endl;
    }
    //return order n
    cout << active.order_of_key(110) << endl;
    return 0;
}
