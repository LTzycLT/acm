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

