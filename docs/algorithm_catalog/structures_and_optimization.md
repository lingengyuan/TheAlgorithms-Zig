# Structures And Optimization / 数据结构与优化方法

- Source of truth: the detailed catalog sections from the pre-split root README.
- 数据来源：拆分前根 README 的详细目录条目。

### Data Structures (101)

| Algorithm | File | Complexity |
|-----------|------|-----------|
| Stack (Array) | [`data_structures/stack.zig`](data_structures/stack.zig) | O(1) amortized push/pop |
| Queue (Array Circular Buffer) | [`data_structures/queue.zig`](data_structures/queue.zig) | O(1) amortized enqueue/dequeue |
| Singly Linked List | [`data_structures/singly_linked_list.zig`](data_structures/singly_linked_list.zig) | O(1) insert head, O(n) insert tail |
| Doubly Linked List | [`data_structures/doubly_linked_list.zig`](data_structures/doubly_linked_list.zig) | O(1) insert/delete head/tail |
| Binary Search Tree | [`data_structures/binary_search_tree.zig`](data_structures/binary_search_tree.zig) | O(log n) avg insert/search/delete |
| Min Heap | [`data_structures/min_heap.zig`](data_structures/min_heap.zig) | O(log n) insert/extract, O(n) heapify |
| Trie | [`data_structures/trie.zig`](data_structures/trie.zig) | O(L) insert/search/delete |
| Disjoint Set (Union-Find) | [`data_structures/disjoint_set.zig`](data_structures/disjoint_set.zig) | O(alpha(n)) amortized |
| AVL Tree | [`data_structures/avl_tree.zig`](data_structures/avl_tree.zig) | O(log n) insert/search/delete |
| Max Heap | [`data_structures/max_heap.zig`](data_structures/max_heap.zig) | O(log n) insert/extract, O(n) heapify |
| Priority Queue | [`data_structures/priority_queue.zig`](data_structures/priority_queue.zig) | O(log n) enqueue/dequeue |
| Hash Map (Open Addressing) | [`data_structures/hash_map_open_addressing.zig`](data_structures/hash_map_open_addressing.zig) | O(1) avg put/get/remove |
| Segment Tree (Range Max Query) | [`data_structures/segment_tree.zig`](data_structures/segment_tree.zig) | O(log n) query/update |
| Fenwick Tree (Binary Indexed Tree) | [`data_structures/fenwick_tree.zig`](data_structures/fenwick_tree.zig) | O(log n) add/prefix/range |
| Red-Black Tree | [`data_structures/red_black_tree.zig`](data_structures/red_black_tree.zig) | O(log n) insert/search |
| LRU Cache | [`data_structures/lru_cache.zig`](data_structures/lru_cache.zig) | O(1) avg get/put |
| Deque (Ring Buffer) | [`data_structures/deque.zig`](data_structures/deque.zig) | O(1) amortized push/pop both ends |
| Sparse Table (RMQ) | [`data_structures/sparse_table.zig`](data_structures/sparse_table.zig) | Build O(n log n), Query O(1) |
| Bloom Filter | [`data_structures/bloom_filter.zig`](data_structures/bloom_filter.zig) | O(k) add/contains, k=2 |
| Circular Linked List | [`data_structures/circular_linked_list.zig`](data_structures/circular_linked_list.zig) | O(1) head/tail, O(n) indexed ops |
| Circular Queue (Fixed Capacity) | [`data_structures/circular_queue.zig`](data_structures/circular_queue.zig) | O(1) enqueue/dequeue |
| Queue by Two Stacks | [`data_structures/queue_by_two_stacks.zig`](data_structures/queue_by_two_stacks.zig) | Amortized O(1) put/get |
| Stack Using Two Queues | [`data_structures/stack_using_two_queues.zig`](data_structures/stack_using_two_queues.zig) | Push O(n), Pop O(1) |
| Treap | [`data_structures/treap.zig`](data_structures/treap.zig) | O(log n) avg insert/search/delete |
| Skip List | [`data_structures/skip_list.zig`](data_structures/skip_list.zig) | O(log n) avg insert/search/delete |
| Linked Queue | [`data_structures/linked_queue.zig`](data_structures/linked_queue.zig) | O(1) put/get |
| Queue by List | [`data_structures/queue_by_list.zig`](data_structures/queue_by_list.zig) | Put O(1), Get O(n) |
| Queue on Pseudo Stack | [`data_structures/queue_on_pseudo_stack.zig`](data_structures/queue_on_pseudo_stack.zig) | Put O(1), Get/Front O(n) |
| Circular Queue (Linked List) | [`data_structures/circular_queue_linked_list.zig`](data_structures/circular_queue_linked_list.zig) | O(1) enqueue/dequeue |
| Priority Queue Using List | [`data_structures/priority_queue_using_list.zig`](data_structures/priority_queue_using_list.zig) | Fixed: O(1) dequeue, Element: O(n) dequeue |
| Stack With Singly Linked List | [`data_structures/stack_with_singly_linked_list.zig`](data_structures/stack_with_singly_linked_list.zig) | O(1) push/pop |
| Stack With Doubly Linked List | [`data_structures/stack_with_doubly_linked_list.zig`](data_structures/stack_with_doubly_linked_list.zig) | O(1) push/pop |
| Deque Doubly (Linked) | [`data_structures/deque_doubly.zig`](data_structures/deque_doubly.zig) | O(1) add/remove both ends |
| Linked List From Sequence | [`data_structures/linked_list_from_sequence.zig`](data_structures/linked_list_from_sequence.zig) | O(n) |
| Middle Element Of Linked List | [`data_structures/middle_element_of_linked_list.zig`](data_structures/middle_element_of_linked_list.zig) | O(n) |
| Linked List Print Reverse | [`data_structures/linked_list_print_reverse.zig`](data_structures/linked_list_print_reverse.zig) | O(n) |
| Linked List Swap Nodes | [`data_structures/linked_list_swap_nodes.zig`](data_structures/linked_list_swap_nodes.zig) | O(n) |
| Linked List Merge Two Lists | [`data_structures/linked_list_merge_two_lists.zig`](data_structures/linked_list_merge_two_lists.zig) | O((n+m) log(n+m)) |
| Linked List Rotate To Right | [`data_structures/linked_list_rotate_to_right.zig`](data_structures/linked_list_rotate_to_right.zig) | O(n) |
| Linked List Palindrome | [`data_structures/linked_list_palindrome.zig`](data_structures/linked_list_palindrome.zig) | O(n) |
| Linked List Has Loop | [`data_structures/linked_list_has_loop.zig`](data_structures/linked_list_has_loop.zig) | O(n) |
| Balanced Parentheses | [`data_structures/balanced_parentheses.zig`](data_structures/balanced_parentheses.zig) | O(n) |
| Next Greater Element | [`data_structures/next_greater_element.zig`](data_structures/next_greater_element.zig) | O(n) |
| Largest Rectangle Histogram | [`data_structures/largest_rectangle_histogram.zig`](data_structures/largest_rectangle_histogram.zig) | O(n) |
| Stock Span Problem | [`data_structures/stock_span_problem.zig`](data_structures/stock_span_problem.zig) | O(n) |
| Postfix Evaluation | [`data_structures/postfix_evaluation.zig`](data_structures/postfix_evaluation.zig) | O(n) |
| Prefix Evaluation | [`data_structures/prefix_evaluation.zig`](data_structures/prefix_evaluation.zig) | O(n) |
| Infix To Postfix Conversion | [`data_structures/infix_to_postfix_conversion.zig`](data_structures/infix_to_postfix_conversion.zig) | O(n) |
| Infix To Prefix Conversion | [`data_structures/infix_to_prefix_conversion.zig`](data_structures/infix_to_prefix_conversion.zig) | O(n) |
| Floyd's Cycle Detection | [`data_structures/floyds_cycle_detection.zig`](data_structures/floyds_cycle_detection.zig) | O(n) |
| Reverse K Group | [`data_structures/reverse_k_group.zig`](data_structures/reverse_k_group.zig) | O(n) |
| Dijkstra's Two-Stack Algorithm | [`data_structures/dijkstras_two_stack_algorithm.zig`](data_structures/dijkstras_two_stack_algorithm.zig) | O(n) |
| Lexicographical Numbers | [`data_structures/lexicographical_numbers.zig`](data_structures/lexicographical_numbers.zig) | O(n) |
| Equilibrium Index In Array | [`data_structures/equilibrium_index_in_array.zig`](data_structures/equilibrium_index_in_array.zig) | O(n) |
| Pairs With Given Sum | [`data_structures/pairs_with_given_sum.zig`](data_structures/pairs_with_given_sum.zig) | O(n) |
| Prefix Sum | [`data_structures/prefix_sum.zig`](data_structures/prefix_sum.zig) | Build O(n), Query O(1) |
| Rotate Array | [`data_structures/rotate_array.zig`](data_structures/rotate_array.zig) | O(n) |
| Monotonic Array Check | [`data_structures/monotonic_array.zig`](data_structures/monotonic_array.zig) | O(n) |
| Kth Largest Element | [`data_structures/kth_largest_element.zig`](data_structures/kth_largest_element.zig) | Average O(n) |
| Median of Two Arrays | [`data_structures/median_two_array.zig`](data_structures/median_two_array.zig) | O((n+m) log(n+m)) |
| Index 2D Array In 1D | [`data_structures/index_2d_array_in_1d.zig`](data_structures/index_2d_array_in_1d.zig) | O(rows) |
| Find Triplets With 0 Sum | [`data_structures/find_triplets_with_0_sum.zig`](data_structures/find_triplets_with_0_sum.zig) | O(n^3) / O(n^2) hashing |
| Permutations (Array Variants) | [`data_structures/permutations.zig`](data_structures/permutations.zig) | O(n! · n) |
| Product Sum (Nested Arrays) | [`data_structures/product_sum.zig`](data_structures/product_sum.zig) | O(n) |
| Double Ended Queue (Linked Nodes) | [`data_structures/double_ended_queue.zig`](data_structures/double_ended_queue.zig) | O(1) append/pop both ends |
| Basic Binary Tree Utilities | [`data_structures/basic_binary_tree.zig`](data_structures/basic_binary_tree.zig) | O(n) traversal/metrics |
| Binary Tree Mirror (Dictionary Form) | [`data_structures/binary_tree_mirror.zig`](data_structures/binary_tree_mirror.zig) | O(n) |
| Binary Tree Node Sum | [`data_structures/binary_tree_node_sum.zig`](data_structures/binary_tree_node_sum.zig) | O(n) |
| Binary Tree Path Sum | [`data_structures/binary_tree_path_sum.zig`](data_structures/binary_tree_path_sum.zig) | O(n²) worst |
| BST Floor And Ceiling | [`data_structures/floor_and_ceiling.zig`](data_structures/floor_and_ceiling.zig) | O(h) |
| Sum Tree Check | [`data_structures/is_sum_tree.zig`](data_structures/is_sum_tree.zig) | O(n) |
| Symmetric Tree Check | [`data_structures/symmetric_tree.zig`](data_structures/symmetric_tree.zig) | O(n) |
| Diameter Of Binary Tree (Node-Centered) | [`data_structures/diameter_of_binary_tree.zig`](data_structures/diameter_of_binary_tree.zig) | O(n) |
| Binary Tree Traversals | [`data_structures/binary_tree_traversals.zig`](data_structures/binary_tree_traversals.zig) | O(n) typical; zigzag O(n²) worst |
| Different Views Of Binary Tree | [`data_structures/diff_views_of_binary_tree.zig`](data_structures/diff_views_of_binary_tree.zig) | O(n log n) |
| Merge Two Binary Trees | [`data_structures/merge_two_binary_trees.zig`](data_structures/merge_two_binary_trees.zig) | O(n) |
| Number Of Possible Binary Trees | [`data_structures/number_of_possible_binary_trees.zig`](data_structures/number_of_possible_binary_trees.zig) | O(n) |
| Serialize/Deserialize Binary Tree | [`data_structures/serialize_deserialize_binary_tree.zig`](data_structures/serialize_deserialize_binary_tree.zig) | O(n) |
| Is Sorted (Local BST Rule) | [`data_structures/is_sorted.zig`](data_structures/is_sorted.zig) | O(n) |
| Mirror Binary Tree | [`data_structures/mirror_binary_tree.zig`](data_structures/mirror_binary_tree.zig) | O(n) |
| Flatten Binary Tree To Linked List | [`data_structures/flatten_binarytree_to_linkedlist.zig`](data_structures/flatten_binarytree_to_linkedlist.zig) | O(n) |
| Distribute Coins In Binary Tree | [`data_structures/distribute_coins.zig`](data_structures/distribute_coins.zig) | O(n) |
| Maximum Sum BST In Binary Tree | [`data_structures/maximum_sum_bst.zig`](data_structures/maximum_sum_bst.zig) | O(n) |
| Inorder Tree Traversal 2022 | [`data_structures/inorder_tree_traversal_2022.zig`](data_structures/inorder_tree_traversal_2022.zig) | Insert O(h), traversal O(n) |
| Binary Search Tree (Recursive) | [`data_structures/binary_search_tree_recursive.zig`](data_structures/binary_search_tree_recursive.zig) | O(h) search/insert/remove |
| Maximum Fenwick Tree | [`data_structures/maximum_fenwick_tree.zig`](data_structures/maximum_fenwick_tree.zig) | O(log² n) update/query |
| Non-Recursive Segment Tree | [`data_structures/non_recursive_segment_tree.zig`](data_structures/non_recursive_segment_tree.zig) | O(log n) update/query |
| Lazy Segment Tree (Range Assign + Max) | [`data_structures/lazy_segment_tree.zig`](data_structures/lazy_segment_tree.zig) | O(log n) update/query |
| Segment Tree (Recursive Node Form) | [`data_structures/segment_tree_other.zig`](data_structures/segment_tree_other.zig) | O(log n) update/query |
| Lowest Common Ancestor (Binary Lifting) | [`data_structures/lowest_common_ancestor.zig`](data_structures/lowest_common_ancestor.zig) | Preprocess O(n log n), query O(log n) |
| Wavelet Tree | [`data_structures/wavelet_tree.zig`](data_structures/wavelet_tree.zig) | Build O(n log sigma), query O(log sigma) |
| Alternate Disjoint Set | [`data_structures/alternate_disjoint_set.zig`](data_structures/alternate_disjoint_set.zig) | Amortized O(alpha(n)) |
| Doubly Linked List (Double Ended Variant) | [`data_structures/doubly_linked_list_two.zig`](data_structures/doubly_linked_list_two.zig) | O(1) head/tail ops, O(n) search |
| Heap (Max Heap) | [`data_structures/heap.zig`](data_structures/heap.zig) | Build O(n), push/pop O(log n) |
| Heap (Generic Item+Score) | [`data_structures/heap_generic.zig`](data_structures/heap_generic.zig) | O(log n) insert/update/delete |
| Skew Heap | [`data_structures/skew_heap.zig`](data_structures/skew_heap.zig) | Amortized O(log n) |
| Randomized Meldable Heap | [`data_structures/randomized_heap.zig`](data_structures/randomized_heap.zig) | Expected O(log n) |
| Hash Table (Linear Probing) | [`data_structures/hash_table.zig`](data_structures/hash_table.zig) | Average O(1) insert/query |
| Hash Table (Linked-List Buckets) | [`data_structures/hash_table_with_linked_list.zig`](data_structures/hash_table_with_linked_list.zig) | Average O(1) insert/query |
| Quadratic Probing Hash Table | [`data_structures/quadratic_probing.zig`](data_structures/quadratic_probing.zig) | Average O(1) insert/query |
| Radix Tree | [`data_structures/radix_tree.zig`](data_structures/radix_tree.zig) | O(L) per operation |

### 数据结构 (101)

| 算法 | 文件 | 复杂度 |
|------|------|--------|
| 栈（数组实现） | [`data_structures/stack.zig`](data_structures/stack.zig) | push/pop 均摊 O(1) |
| 队列（数组环形缓冲区） | [`data_structures/queue.zig`](data_structures/queue.zig) | enqueue/dequeue 均摊 O(1) |
| 单向链表 | [`data_structures/singly_linked_list.zig`](data_structures/singly_linked_list.zig) | 头插 O(1)，尾插 O(n) |
| 双向链表 | [`data_structures/doubly_linked_list.zig`](data_structures/doubly_linked_list.zig) | 头尾插删 O(1) |
| 二叉搜索树 | [`data_structures/binary_search_tree.zig`](data_structures/binary_search_tree.zig) | 插入/查找/删除平均 O(log n) |
| 最小堆 | [`data_structures/min_heap.zig`](data_structures/min_heap.zig) | 插入/取出 O(log n)，建堆 O(n) |
| Trie（前缀树） | [`data_structures/trie.zig`](data_structures/trie.zig) | 插入/查询/删除 O(L) |
| 并查集（Union-Find） | [`data_structures/disjoint_set.zig`](data_structures/disjoint_set.zig) | 均摊 O(alpha(n)) |
| AVL 树 | [`data_structures/avl_tree.zig`](data_structures/avl_tree.zig) | 插入/查找/删除 O(log n) |
| 最大堆 | [`data_structures/max_heap.zig`](data_structures/max_heap.zig) | 插入/取出 O(log n)，建堆 O(n) |
| 优先队列 | [`data_structures/priority_queue.zig`](data_structures/priority_queue.zig) | 入队/出队 O(log n) |
| 开放寻址哈希表 | [`data_structures/hash_map_open_addressing.zig`](data_structures/hash_map_open_addressing.zig) | put/get/remove 平均 O(1) |
| 线段树（区间最大值） | [`data_structures/segment_tree.zig`](data_structures/segment_tree.zig) | 查询/更新 O(log n) |
| 树状数组（Fenwick Tree） | [`data_structures/fenwick_tree.zig`](data_structures/fenwick_tree.zig) | add/prefix/range O(log n) |
| 红黑树 | [`data_structures/red_black_tree.zig`](data_structures/red_black_tree.zig) | 插入/查找 O(log n) |
| LRU 缓存 | [`data_structures/lru_cache.zig`](data_structures/lru_cache.zig) | get/put 平均 O(1) |
| 双端队列（环形缓冲） | [`data_structures/deque.zig`](data_structures/deque.zig) | 两端 push/pop 均摊 O(1) |
| 稀疏表（RMQ） | [`data_structures/sparse_table.zig`](data_structures/sparse_table.zig) | 构建 O(n log n)，查询 O(1) |
| 布隆过滤器 | [`data_structures/bloom_filter.zig`](data_structures/bloom_filter.zig) | add/contains 为 O(k)，k=2 |
| 循环链表 | [`data_structures/circular_linked_list.zig`](data_structures/circular_linked_list.zig) | 头尾操作 O(1)，按索引 O(n) |
| 循环队列（定长） | [`data_structures/circular_queue.zig`](data_structures/circular_queue.zig) | enqueue/dequeue O(1) |
| 双栈实现队列 | [`data_structures/queue_by_two_stacks.zig`](data_structures/queue_by_two_stacks.zig) | put/get 均摊 O(1) |
| 双队列实现栈 | [`data_structures/stack_using_two_queues.zig`](data_structures/stack_using_two_queues.zig) | push O(n)，pop O(1) |
| Treap（树堆） | [`data_structures/treap.zig`](data_structures/treap.zig) | 插入/查找/删除平均 O(log n) |
| 跳表 | [`data_structures/skip_list.zig`](data_structures/skip_list.zig) | 插入/查找/删除平均 O(log n) |
| 链式队列 | [`data_structures/linked_queue.zig`](data_structures/linked_queue.zig) | put/get O(1) |
| 列表实现队列 | [`data_structures/queue_by_list.zig`](data_structures/queue_by_list.zig) | put O(1)，get O(n) |
| 伪栈实现队列 | [`data_structures/queue_on_pseudo_stack.zig`](data_structures/queue_on_pseudo_stack.zig) | put O(1)，get/front O(n) |
| 循环队列（链表） | [`data_structures/circular_queue_linked_list.zig`](data_structures/circular_queue_linked_list.zig) | enqueue/dequeue O(1) |
| 列表实现优先队列 | [`data_structures/priority_queue_using_list.zig`](data_structures/priority_queue_using_list.zig) | 固定优先级出队 O(1)，元素优先级出队 O(n) |
| 单链表实现栈 | [`data_structures/stack_with_singly_linked_list.zig`](data_structures/stack_with_singly_linked_list.zig) | push/pop O(1) |
| 双链表实现栈 | [`data_structures/stack_with_doubly_linked_list.zig`](data_structures/stack_with_doubly_linked_list.zig) | push/pop O(1) |
| 双向链表双端队列 | [`data_structures/deque_doubly.zig`](data_structures/deque_doubly.zig) | 两端 add/remove O(1) |
| 序列构建链表 | [`data_structures/linked_list_from_sequence.zig`](data_structures/linked_list_from_sequence.zig) | O(n) |
| 链表中间元素 | [`data_structures/middle_element_of_linked_list.zig`](data_structures/middle_element_of_linked_list.zig) | O(n) |
| 链表逆序输出 | [`data_structures/linked_list_print_reverse.zig`](data_structures/linked_list_print_reverse.zig) | O(n) |
| 链表节点交换 | [`data_structures/linked_list_swap_nodes.zig`](data_structures/linked_list_swap_nodes.zig) | O(n) |
| 合并两个有序链表 | [`data_structures/linked_list_merge_two_lists.zig`](data_structures/linked_list_merge_two_lists.zig) | O((n+m) log(n+m)) |
| 链表右旋 | [`data_structures/linked_list_rotate_to_right.zig`](data_structures/linked_list_rotate_to_right.zig) | O(n) |
| 链表回文判断 | [`data_structures/linked_list_palindrome.zig`](data_structures/linked_list_palindrome.zig) | O(n) |
| 链表环检测 | [`data_structures/linked_list_has_loop.zig`](data_structures/linked_list_has_loop.zig) | O(n) |
| 括号平衡检查 | [`data_structures/balanced_parentheses.zig`](data_structures/balanced_parentheses.zig) | O(n) |
| 下一个更大元素 | [`data_structures/next_greater_element.zig`](data_structures/next_greater_element.zig) | O(n) |
| 柱状图最大矩形 | [`data_structures/largest_rectangle_histogram.zig`](data_structures/largest_rectangle_histogram.zig) | O(n) |
| 股票跨度问题 | [`data_structures/stock_span_problem.zig`](data_structures/stock_span_problem.zig) | O(n) |
| 后缀表达式求值 | [`data_structures/postfix_evaluation.zig`](data_structures/postfix_evaluation.zig) | O(n) |
| 前缀表达式求值 | [`data_structures/prefix_evaluation.zig`](data_structures/prefix_evaluation.zig) | O(n) |
| 中缀转后缀 | [`data_structures/infix_to_postfix_conversion.zig`](data_structures/infix_to_postfix_conversion.zig) | O(n) |
| 中缀转前缀 | [`data_structures/infix_to_prefix_conversion.zig`](data_structures/infix_to_prefix_conversion.zig) | O(n) |
| Floyd 判圈算法 | [`data_structures/floyds_cycle_detection.zig`](data_structures/floyds_cycle_detection.zig) | O(n) |
| K 组链表反转 | [`data_structures/reverse_k_group.zig`](data_structures/reverse_k_group.zig) | O(n) |
| Dijkstra 双栈求值 | [`data_structures/dijkstras_two_stack_algorithm.zig`](data_structures/dijkstras_two_stack_algorithm.zig) | O(n) |
| 字典序数字生成 | [`data_structures/lexicographical_numbers.zig`](data_structures/lexicographical_numbers.zig) | O(n) |
| 数组平衡下标 | [`data_structures/equilibrium_index_in_array.zig`](data_structures/equilibrium_index_in_array.zig) | O(n) |
| 指定和配对计数 | [`data_structures/pairs_with_given_sum.zig`](data_structures/pairs_with_given_sum.zig) | O(n) |
| 前缀和 | [`data_structures/prefix_sum.zig`](data_structures/prefix_sum.zig) | 构建 O(n)，查询 O(1) |
| 数组旋转 | [`data_structures/rotate_array.zig`](data_structures/rotate_array.zig) | O(n) |
| 单调数组检查 | [`data_structures/monotonic_array.zig`](data_structures/monotonic_array.zig) | O(n) |
| 第 k 大元素 | [`data_structures/kth_largest_element.zig`](data_structures/kth_largest_element.zig) | 平均 O(n) |
| 两数组中位数 | [`data_structures/median_two_array.zig`](data_structures/median_two_array.zig) | O((n+m) log(n+m)) |
| 二维数组一维索引 | [`data_structures/index_2d_array_in_1d.zig`](data_structures/index_2d_array_in_1d.zig) | O(rows) |
| 零和三元组 | [`data_structures/find_triplets_with_0_sum.zig`](data_structures/find_triplets_with_0_sum.zig) | O(n^3) / 哈希 O(n^2) |
| 全排列（数组版本） | [`data_structures/permutations.zig`](data_structures/permutations.zig) | O(n! · n) |
| 嵌套数组乘积和 | [`data_structures/product_sum.zig`](data_structures/product_sum.zig) | O(n) |
| 双端队列（双向链表节点） | [`data_structures/double_ended_queue.zig`](data_structures/double_ended_queue.zig) | 两端 append/pop O(1) |
| 基础二叉树工具集 | [`data_structures/basic_binary_tree.zig`](data_structures/basic_binary_tree.zig) | 遍历/度量 O(n) |
| 二叉树镜像（字典表示） | [`data_structures/binary_tree_mirror.zig`](data_structures/binary_tree_mirror.zig) | O(n) |
| 二叉树节点求和 | [`data_structures/binary_tree_node_sum.zig`](data_structures/binary_tree_node_sum.zig) | O(n) |
| 二叉树路径和计数 | [`data_structures/binary_tree_path_sum.zig`](data_structures/binary_tree_path_sum.zig) | 最坏 O(n²) |
| BST Floor / Ceiling | [`data_structures/floor_and_ceiling.zig`](data_structures/floor_and_ceiling.zig) | O(h) |
| Sum Tree 判定 | [`data_structures/is_sum_tree.zig`](data_structures/is_sum_tree.zig) | O(n) |
| 对称二叉树判定 | [`data_structures/symmetric_tree.zig`](data_structures/symmetric_tree.zig) | O(n) |
| 二叉树直径（节点中心定义） | [`data_structures/diameter_of_binary_tree.zig`](data_structures/diameter_of_binary_tree.zig) | O(n) |
| 二叉树遍历集合 | [`data_structures/binary_tree_traversals.zig`](data_structures/binary_tree_traversals.zig) | 常规 O(n)，zigzag 最坏 O(n²) |
| 二叉树多视图（左/右/上/下） | [`data_structures/diff_views_of_binary_tree.zig`](data_structures/diff_views_of_binary_tree.zig) | O(n log n) |
| 合并两棵二叉树 | [`data_structures/merge_two_binary_trees.zig`](data_structures/merge_two_binary_trees.zig) | O(n) |
| 二叉树数量（Catalan/总数） | [`data_structures/number_of_possible_binary_trees.zig`](data_structures/number_of_possible_binary_trees.zig) | O(n) |
| 二叉树序列化/反序列化 | [`data_structures/serialize_deserialize_binary_tree.zig`](data_structures/serialize_deserialize_binary_tree.zig) | O(n) |
| 有序性检查（局部 BST 规则） | [`data_structures/is_sorted.zig`](data_structures/is_sorted.zig) | O(n) |
| 二叉树镜像 | [`data_structures/mirror_binary_tree.zig`](data_structures/mirror_binary_tree.zig) | O(n) |
| 二叉树拍平为链表 | [`data_structures/flatten_binarytree_to_linkedlist.zig`](data_structures/flatten_binarytree_to_linkedlist.zig) | O(n) |
| 二叉树硬币分配 | [`data_structures/distribute_coins.zig`](data_structures/distribute_coins.zig) | O(n) |
| 二叉树中 BST 子树最大和 | [`data_structures/maximum_sum_bst.zig`](data_structures/maximum_sum_bst.zig) | O(n) |
| 中序遍历（2022 版本） | [`data_structures/inorder_tree_traversal_2022.zig`](data_structures/inorder_tree_traversal_2022.zig) | 插入 O(h)，遍历 O(n) |
| 二叉搜索树（递归实现） | [`data_structures/binary_search_tree_recursive.zig`](data_structures/binary_search_tree_recursive.zig) | 搜索/插入/删除 O(h) |
| 最大值 Fenwick 树 | [`data_structures/maximum_fenwick_tree.zig`](data_structures/maximum_fenwick_tree.zig) | 更新/查询 O(log² n) |
| 非递归线段树 | [`data_structures/non_recursive_segment_tree.zig`](data_structures/non_recursive_segment_tree.zig) | 更新/查询 O(log n) |
| 懒标记线段树（区间赋值+最大值） | [`data_structures/lazy_segment_tree.zig`](data_structures/lazy_segment_tree.zig) | 更新/查询 O(log n) |
| 线段树（递归节点实现） | [`data_structures/segment_tree_other.zig`](data_structures/segment_tree_other.zig) | 更新/查询 O(log n) |
| 最近公共祖先（倍增法） | [`data_structures/lowest_common_ancestor.zig`](data_structures/lowest_common_ancestor.zig) | 预处理 O(n log n)，查询 O(log n) |
| Wavelet 树 | [`data_structures/wavelet_tree.zig`](data_structures/wavelet_tree.zig) | 构建 O(n log sigma)，查询 O(log sigma) |
| 并查集（替代实现） | [`data_structures/alternate_disjoint_set.zig`](data_structures/alternate_disjoint_set.zig) | 均摊 O(alpha(n)) |
| 双向链表（双端版本） | [`data_structures/doubly_linked_list_two.zig`](data_structures/doubly_linked_list_two.zig) | 头尾 O(1)，查找 O(n) |
| 堆（最大堆） | [`data_structures/heap.zig`](data_structures/heap.zig) | 构建 O(n)，插入/弹出 O(log n) |
| 堆（通用 item+score） | [`data_structures/heap_generic.zig`](data_structures/heap_generic.zig) | 插入/更新/删除 O(log n) |
| 斜堆 | [`data_structures/skew_heap.zig`](data_structures/skew_heap.zig) | 均摊 O(log n) |
| 随机可并堆 | [`data_structures/randomized_heap.zig`](data_structures/randomized_heap.zig) | 期望 O(log n) |
| 哈希表（线性探测） | [`data_structures/hash_table.zig`](data_structures/hash_table.zig) | 平均 O(1) 插入/查询 |
| 哈希表（链表桶） | [`data_structures/hash_table_with_linked_list.zig`](data_structures/hash_table_with_linked_list.zig) | 平均 O(1) 插入/查询 |
| 哈希表（二次探测） | [`data_structures/quadratic_probing.zig`](data_structures/quadratic_probing.zig) | 平均 O(1) 插入/查询 |
| Radix 树（压缩前缀树） | [`data_structures/radix_tree.zig`](data_structures/radix_tree.zig) | 每次操作 O(L) |

### Dynamic Programming (54)

| Algorithm | File | Complexity |
|-----------|------|-----------|
| Climbing Stairs | [`dynamic_programming/climbing_stairs.zig`](dynamic_programming/climbing_stairs.zig) | O(n) |
| Fibonacci (Memoized DP) | [`dynamic_programming/fibonacci_dp.zig`](dynamic_programming/fibonacci_dp.zig) | O(n) |
| Fibonacci (Compatibility Wrapper) | [`dynamic_programming/fibonacci.zig`](dynamic_programming/fibonacci.zig) | O(n) |
| Coin Change (Ways) | [`dynamic_programming/coin_change.zig`](dynamic_programming/coin_change.zig) | O(amount × coin_count) |
| Minimum Coin Change (Compatibility Wrapper) | [`dynamic_programming/minimum_coin_change.zig`](dynamic_programming/minimum_coin_change.zig) | O(amount × coin_count) |
| All Construct | [`dynamic_programming/all_construct.zig`](dynamic_programming/all_construct.zig) | Output-sensitive exponential |
| Factorial (DP Compatibility Variant) | [`dynamic_programming/factorial.zig`](dynamic_programming/factorial.zig) | O(n) |
| Max Subarray Sum (Kadane) | [`dynamic_programming/max_subarray_sum.zig`](dynamic_programming/max_subarray_sum.zig) | O(n) |
| Longest Increasing Subsequence | [`dynamic_programming/longest_increasing_subsequence.zig`](dynamic_programming/longest_increasing_subsequence.zig) | O(n log n) |
| Rod Cutting | [`dynamic_programming/rod_cutting.zig`](dynamic_programming/rod_cutting.zig) | O(n²) |
| Matrix Chain Multiplication | [`dynamic_programming/matrix_chain_multiplication.zig`](dynamic_programming/matrix_chain_multiplication.zig) | O(n³) |
| Palindrome Partitioning (Min Cuts) | [`dynamic_programming/palindrome_partitioning.zig`](dynamic_programming/palindrome_partitioning.zig) | O(n²) |
| Word Break | [`dynamic_programming/word_break.zig`](dynamic_programming/word_break.zig) | O(n·m·k) |
| Catalan Numbers | [`dynamic_programming/catalan_numbers.zig`](dynamic_programming/catalan_numbers.zig) | O(n²) |
| Longest Common Subsequence | [`dynamic_programming/longest_common_subsequence.zig`](dynamic_programming/longest_common_subsequence.zig) | O(m × n) |
| Edit Distance | [`dynamic_programming/edit_distance.zig`](dynamic_programming/edit_distance.zig) | O(m × n) |
| 0/1 Knapsack | [`dynamic_programming/knapsack.zig`](dynamic_programming/knapsack.zig) | O(n × W) |
| Subset Sum | [`dynamic_programming/subset_sum.zig`](dynamic_programming/subset_sum.zig) | O(n × target) |
| Sum Of Subset (Compatibility Wrapper) | [`dynamic_programming/sum_of_subset.zig`](dynamic_programming/sum_of_subset.zig) | O(n × target) |
| Egg Drop Problem | [`dynamic_programming/egg_drop_problem.zig`](dynamic_programming/egg_drop_problem.zig) | O(eggs × answer) |
| Longest Palindromic Subsequence | [`dynamic_programming/longest_palindromic_subsequence.zig`](dynamic_programming/longest_palindromic_subsequence.zig) | O(n²) |
| Maximum Product Subarray | [`dynamic_programming/max_product_subarray.zig`](dynamic_programming/max_product_subarray.zig) | O(n) |
| Combination Sum IV (Ordered Combinations) | [`dynamic_programming/combination_sum_iv.zig`](dynamic_programming/combination_sum_iv.zig) | O(target × n) |
| Minimum Steps to One | [`dynamic_programming/min_steps_to_one.zig`](dynamic_programming/min_steps_to_one.zig) | O(n) |
| Minimum Steps to One (Compatibility Wrapper) | [`dynamic_programming/minimum_steps_to_one.zig`](dynamic_programming/minimum_steps_to_one.zig) | O(n) |
| Minimum Cost Path (Grid) | [`dynamic_programming/minimum_cost_path.zig`](dynamic_programming/minimum_cost_path.zig) | O(rows × cols) |
| Minimum Tickets Cost | [`dynamic_programming/minimum_tickets_cost.zig`](dynamic_programming/minimum_tickets_cost.zig) | O(365) |
| Regex Match (`.` and `*`) | [`dynamic_programming/regex_match.zig`](dynamic_programming/regex_match.zig) | O(m × n) |
| Smith-Waterman Local Alignment | [`dynamic_programming/smith_waterman.zig`](dynamic_programming/smith_waterman.zig) | O(m × n) |
| Subset Generation (n-combinations) | [`dynamic_programming/subset_generation.zig`](dynamic_programming/subset_generation.zig) | O(C(r, n) × n) |
| Wildcard Matching (`?` and `*`) | [`dynamic_programming/wildcard_matching.zig`](dynamic_programming/wildcard_matching.zig) | O(m × n) |
| Integer Partition Count | [`dynamic_programming/integer_partition.zig`](dynamic_programming/integer_partition.zig) | O(n²) |
| Tribonacci Sequence | [`dynamic_programming/tribonacci.zig`](dynamic_programming/tribonacci.zig) | O(n) |
| Maximum Non-Adjacent Sum | [`dynamic_programming/max_non_adjacent_sum.zig`](dynamic_programming/max_non_adjacent_sum.zig) | O(n) |
| Minimum Partition Difference | [`dynamic_programming/minimum_partition.zig`](dynamic_programming/minimum_partition.zig) | O(n × total_sum) |
| Minimum Squares To Represent A Number | [`dynamic_programming/minimum_squares_to_represent_a_number.zig`](dynamic_programming/minimum_squares_to_represent_a_number.zig) | O(n × sqrt(n)) |
| Longest Common Substring | [`dynamic_programming/longest_common_substring.zig`](dynamic_programming/longest_common_substring.zig) | O(m × n) |
| Largest Divisible Subset | [`dynamic_programming/largest_divisible_subset.zig`](dynamic_programming/largest_divisible_subset.zig) | O(n²) |
| Optimal Binary Search Tree (Cost) | [`dynamic_programming/optimal_binary_search_tree.zig`](dynamic_programming/optimal_binary_search_tree.zig) | O(n²) |
| Range Sum Query (Prefix Sum) | [`dynamic_programming/range_sum_query.zig`](dynamic_programming/range_sum_query.zig) | O(n + q) |
| Minimum Size Subarray Sum | [`dynamic_programming/minimum_size_subarray_sum.zig`](dynamic_programming/minimum_size_subarray_sum.zig) | O(n) |
| Abbreviation DP | [`dynamic_programming/abbreviation.zig`](dynamic_programming/abbreviation.zig) | O(n × m) |
| Matrix Chain Order (Cost + Split Tables) | [`dynamic_programming/matrix_chain_order.zig`](dynamic_programming/matrix_chain_order.zig) | O(n³) |
| Min Distance Up-Bottom (Top-Down Edit Distance) | [`dynamic_programming/min_distance_up_bottom.zig`](dynamic_programming/min_distance_up_bottom.zig) | O(m × n) |
| Floyd-Warshall (Graph Wrapper) | [`dynamic_programming/floyd_warshall.zig`](dynamic_programming/floyd_warshall.zig) | O(n³) |
| Narcissistic Number Search | [`dynamic_programming/narcissistic_number.zig`](dynamic_programming/narcissistic_number.zig) | O(limit × digits) |
| Trapped Rainwater | [`dynamic_programming/trapped_water.zig`](dynamic_programming/trapped_water.zig) | O(n) |
| Iterating Through Submasks | [`dynamic_programming/iterating_through_submasks.zig`](dynamic_programming/iterating_through_submasks.zig) | O(2^k) |
| Fast Fibonacci (Doubling) | [`dynamic_programming/fast_fibonacci.zig`](dynamic_programming/fast_fibonacci.zig) | O(log n) |
| Fizz Buzz | [`dynamic_programming/fizz_buzz.zig`](dynamic_programming/fizz_buzz.zig) | O(iterations) |
| LIS Iterative (Sequence, O(n²)) | [`dynamic_programming/longest_increasing_subsequence_iterative.zig`](dynamic_programming/longest_increasing_subsequence_iterative.zig) | O(n²) |
| LIS Length (O(n log n)) | [`dynamic_programming/longest_increasing_subsequence_o_nlogn.zig`](dynamic_programming/longest_increasing_subsequence_o_nlogn.zig) | O(n log n) |
| Viterbi Algorithm | [`dynamic_programming/viterbi.zig`](dynamic_programming/viterbi.zig) | O(T × S²) |
| Assignment Using Bitmask | [`dynamic_programming/bitmask.zig`](dynamic_programming/bitmask.zig) | O(2^P · T · avg_deg) |

### 动态规划 (54)

| 算法 | 文件 | 复杂度 |
|------|------|--------|
| 爬楼梯 | [`dynamic_programming/climbing_stairs.zig`](dynamic_programming/climbing_stairs.zig) | O(n) |
| 斐波那契（记忆化 DP） | [`dynamic_programming/fibonacci_dp.zig`](dynamic_programming/fibonacci_dp.zig) | O(n) |
| 斐波那契（兼容包装） | [`dynamic_programming/fibonacci.zig`](dynamic_programming/fibonacci.zig) | O(n) |
| 硬币找零（方案数） | [`dynamic_programming/coin_change.zig`](dynamic_programming/coin_change.zig) | O(amount × coin_count) |
| 最少硬币找零（兼容包装） | [`dynamic_programming/minimum_coin_change.zig`](dynamic_programming/minimum_coin_change.zig) | O(amount × coin_count) |
| 全部构造方案 | [`dynamic_programming/all_construct.zig`](dynamic_programming/all_construct.zig) | 按输出规模指数增长 |
| 阶乘（DP 兼容版本） | [`dynamic_programming/factorial.zig`](dynamic_programming/factorial.zig) | O(n) |
| 最大子数组和（Kadane） | [`dynamic_programming/max_subarray_sum.zig`](dynamic_programming/max_subarray_sum.zig) | O(n) |
| 最长递增子序列 | [`dynamic_programming/longest_increasing_subsequence.zig`](dynamic_programming/longest_increasing_subsequence.zig) | O(n log n) |
| 钢条切割 | [`dynamic_programming/rod_cutting.zig`](dynamic_programming/rod_cutting.zig) | O(n²) |
| 矩阵链乘法 | [`dynamic_programming/matrix_chain_multiplication.zig`](dynamic_programming/matrix_chain_multiplication.zig) | O(n³) |
| 回文划分（最少切割） | [`dynamic_programming/palindrome_partitioning.zig`](dynamic_programming/palindrome_partitioning.zig) | O(n²) |
| 单词拆分 | [`dynamic_programming/word_break.zig`](dynamic_programming/word_break.zig) | O(n·m·k) |
| Catalan 数 | [`dynamic_programming/catalan_numbers.zig`](dynamic_programming/catalan_numbers.zig) | O(n²) |
| 最长公共子序列 | [`dynamic_programming/longest_common_subsequence.zig`](dynamic_programming/longest_common_subsequence.zig) | O(m × n) |
| 编辑距离 | [`dynamic_programming/edit_distance.zig`](dynamic_programming/edit_distance.zig) | O(m × n) |
| 0/1 背包 | [`dynamic_programming/knapsack.zig`](dynamic_programming/knapsack.zig) | O(n × W) |
| 子集和 | [`dynamic_programming/subset_sum.zig`](dynamic_programming/subset_sum.zig) | O(n × target) |
| 子集求和（兼容包装） | [`dynamic_programming/sum_of_subset.zig`](dynamic_programming/sum_of_subset.zig) | O(n × target) |
| 鸡蛋掉落问题 | [`dynamic_programming/egg_drop_problem.zig`](dynamic_programming/egg_drop_problem.zig) | O(eggs × answer) |
| 最长回文子序列 | [`dynamic_programming/longest_palindromic_subsequence.zig`](dynamic_programming/longest_palindromic_subsequence.zig) | O(n²) |
| 最大乘积子数组 | [`dynamic_programming/max_product_subarray.zig`](dynamic_programming/max_product_subarray.zig) | O(n) |
| 组合总和 IV（有序方案数） | [`dynamic_programming/combination_sum_iv.zig`](dynamic_programming/combination_sum_iv.zig) | O(target × n) |
| 到 1 的最少步数 | [`dynamic_programming/min_steps_to_one.zig`](dynamic_programming/min_steps_to_one.zig) | O(n) |
| 到 1 的最少步数（兼容包装） | [`dynamic_programming/minimum_steps_to_one.zig`](dynamic_programming/minimum_steps_to_one.zig) | O(n) |
| 最小路径代价（网格） | [`dynamic_programming/minimum_cost_path.zig`](dynamic_programming/minimum_cost_path.zig) | O(rows × cols) |
| 最低票价 | [`dynamic_programming/minimum_tickets_cost.zig`](dynamic_programming/minimum_tickets_cost.zig) | O(365) |
| 正则匹配（`.` 与 `*`） | [`dynamic_programming/regex_match.zig`](dynamic_programming/regex_match.zig) | O(m × n) |
| Smith-Waterman 局部序列比对 | [`dynamic_programming/smith_waterman.zig`](dynamic_programming/smith_waterman.zig) | O(m × n) |
| 子集生成（n 元组合） | [`dynamic_programming/subset_generation.zig`](dynamic_programming/subset_generation.zig) | O(C(r, n) × n) |
| 通配符匹配（`?` 与 `*`） | [`dynamic_programming/wildcard_matching.zig`](dynamic_programming/wildcard_matching.zig) | O(m × n) |
| 整数拆分计数 | [`dynamic_programming/integer_partition.zig`](dynamic_programming/integer_partition.zig) | O(n²) |
| Tribonacci 数列 | [`dynamic_programming/tribonacci.zig`](dynamic_programming/tribonacci.zig) | O(n) |
| 最大非相邻子序列和 | [`dynamic_programming/max_non_adjacent_sum.zig`](dynamic_programming/max_non_adjacent_sum.zig) | O(n) |
| 最小划分差值 | [`dynamic_programming/minimum_partition.zig`](dynamic_programming/minimum_partition.zig) | O(n × total_sum) |
| 表示为平方数和的最少项数 | [`dynamic_programming/minimum_squares_to_represent_a_number.zig`](dynamic_programming/minimum_squares_to_represent_a_number.zig) | O(n × sqrt(n)) |
| 最长公共子串 | [`dynamic_programming/longest_common_substring.zig`](dynamic_programming/longest_common_substring.zig) | O(m × n) |
| 最大可整除子集 | [`dynamic_programming/largest_divisible_subset.zig`](dynamic_programming/largest_divisible_subset.zig) | O(n²) |
| 最优二叉搜索树（代价） | [`dynamic_programming/optimal_binary_search_tree.zig`](dynamic_programming/optimal_binary_search_tree.zig) | O(n²) |
| 区间和查询（前缀和） | [`dynamic_programming/range_sum_query.zig`](dynamic_programming/range_sum_query.zig) | O(n + q) |
| 最短满足和子数组长度 | [`dynamic_programming/minimum_size_subarray_sum.zig`](dynamic_programming/minimum_size_subarray_sum.zig) | O(n) |
| 字符串缩写匹配 DP | [`dynamic_programming/abbreviation.zig`](dynamic_programming/abbreviation.zig) | O(n × m) |
| 矩阵链次序（代价+切分表） | [`dynamic_programming/matrix_chain_order.zig`](dynamic_programming/matrix_chain_order.zig) | O(n³) |
| 编辑距离上自顶向下版 | [`dynamic_programming/min_distance_up_bottom.zig`](dynamic_programming/min_distance_up_bottom.zig) | O(m × n) |
| Floyd-Warshall（图包装） | [`dynamic_programming/floyd_warshall.zig`](dynamic_programming/floyd_warshall.zig) | O(n³) |
| 自恋数搜索 | [`dynamic_programming/narcissistic_number.zig`](dynamic_programming/narcissistic_number.zig) | O(limit × digits) |
| 接雨水 | [`dynamic_programming/trapped_water.zig`](dynamic_programming/trapped_water.zig) | O(n) |
| 子掩码遍历 | [`dynamic_programming/iterating_through_submasks.zig`](dynamic_programming/iterating_through_submasks.zig) | O(2^k) |
| 快速斐波那契（倍增法） | [`dynamic_programming/fast_fibonacci.zig`](dynamic_programming/fast_fibonacci.zig) | O(log n) |
| Fizz Buzz | [`dynamic_programming/fizz_buzz.zig`](dynamic_programming/fizz_buzz.zig) | O(iterations) |
| 最长递增子序列迭代版（序列，O(n²)） | [`dynamic_programming/longest_increasing_subsequence_iterative.zig`](dynamic_programming/longest_increasing_subsequence_iterative.zig) | O(n²) |
| 最长递增子序列长度（O(n log n)） | [`dynamic_programming/longest_increasing_subsequence_o_nlogn.zig`](dynamic_programming/longest_increasing_subsequence_o_nlogn.zig) | O(n log n) |
| Viterbi 算法 | [`dynamic_programming/viterbi.zig`](dynamic_programming/viterbi.zig) | O(T × S²) |
| Bitmask 任务分配计数 | [`dynamic_programming/bitmask.zig`](dynamic_programming/bitmask.zig) | O(2^P · T · avg_deg) |

### Graphs (82)

| Algorithm | File | Complexity |
|-----------|------|-----------|
| Breadth-First Search (BFS) | [`graphs/bfs.zig`](graphs/bfs.zig) | O(V + E) |
| Depth-First Search (DFS) | [`graphs/dfs.zig`](graphs/dfs.zig) | O(V + E) |
| Dijkstra Shortest Path | [`graphs/dijkstra.zig`](graphs/dijkstra.zig) | O(V² + E) |
| A* Search | [`graphs/a_star_search.zig`](graphs/a_star_search.zig) | O(V² + E) |
| Tarjan SCC | [`graphs/tarjan_scc.zig`](graphs/tarjan_scc.zig) | O(V + E) |
| Bridges (Articulation Edges) | [`graphs/bridges.zig`](graphs/bridges.zig) | O(V + E) |
| Eulerian Path/Circuit (Undirected) | [`graphs/eulerian_path_circuit_undirected.zig`](graphs/eulerian_path_circuit_undirected.zig) | O(V + E) |
| Ford-Fulkerson Max Flow | [`graphs/ford_fulkerson.zig`](graphs/ford_fulkerson.zig) | O(V · E²) |
| Bipartite Check (BFS) | [`graphs/bipartite_check_bfs.zig`](graphs/bipartite_check_bfs.zig) | O(V + E) |
| Bellman-Ford Shortest Path | [`graphs/bellman_ford.zig`](graphs/bellman_ford.zig) | O(V·E) |
| Topological Sort | [`graphs/topological_sort.zig`](graphs/topological_sort.zig) | O(V + E) |
| Floyd-Warshall | [`graphs/floyd_warshall.zig`](graphs/floyd_warshall.zig) | O(V³) |
| Detect Cycle (Directed) | [`graphs/detect_cycle.zig`](graphs/detect_cycle.zig) | O(V + E) |
| Connected Components | [`graphs/connected_components.zig`](graphs/connected_components.zig) | O(V + E) |
| Kruskal MST | [`graphs/kruskal.zig`](graphs/kruskal.zig) | O(E log E) |
| Prim MST | [`graphs/prim.zig`](graphs/prim.zig) | O(V² + E) |
| Articulation Points | [`graphs/articulation_points.zig`](graphs/articulation_points.zig) | O(V + E) |
| Kosaraju SCC | [`graphs/kosaraju_scc.zig`](graphs/kosaraju_scc.zig) | O(V + E) |
| Kahn Topological Sort | [`graphs/kahn_topological_sort.zig`](graphs/kahn_topological_sort.zig) | O(V + E) |
| BFS Shortest Path | [`graphs/breadth_first_search_shortest_path.zig`](graphs/breadth_first_search_shortest_path.zig) | O(V + E) |
| Boruvka MST | [`graphs/boruvka_mst.zig`](graphs/boruvka_mst.zig) | O(E log V) |
| 0-1 BFS Shortest Path | [`graphs/zero_one_bfs_shortest_path.zig`](graphs/zero_one_bfs_shortest_path.zig) | O(V + E) |
| Bidirectional BFS Path | [`graphs/bidirectional_breadth_first_search.zig`](graphs/bidirectional_breadth_first_search.zig) | O(V + E) |
| Dijkstra on Binary Grid | [`graphs/dijkstra_binary_grid.zig`](graphs/dijkstra_binary_grid.zig) | O((R·C)²) |
| Even Tree | [`graphs/even_tree.zig`](graphs/even_tree.zig) | O(V + E) |
| Gale-Shapley Stable Matching | [`graphs/gale_shapley_stable_matching.zig`](graphs/gale_shapley_stable_matching.zig) | O(n²) |
| PageRank (Iterative) | [`graphs/page_rank.zig`](graphs/page_rank.zig) | O(iterations·(V + E)) |
| Bidirectional Dijkstra | [`graphs/bidirectional_dijkstra.zig`](graphs/bidirectional_dijkstra.zig) | O(V² + E) |
| Greedy Best-First Search | [`graphs/greedy_best_first.zig`](graphs/greedy_best_first.zig) | O(V²) |
| Dinic Max Flow | [`graphs/dinic_max_flow.zig`](graphs/dinic_max_flow.zig) | O(V²·E) |
| Bidirectional Search | [`graphs/bidirectional_search.zig`](graphs/bidirectional_search.zig) | O(V + E) |
| Minimum Path Sum | [`graphs/minimum_path_sum.zig`](graphs/minimum_path_sum.zig) | O(rows·cols) |
| Deep Clone Graph | [`graphs/deep_clone_graph.zig`](graphs/deep_clone_graph.zig) | O(V + E) |
| Dijkstra (Matrix) | [`graphs/dijkstra_matrix.zig`](graphs/dijkstra_matrix.zig) | O(V²) |
| Breadth-First Search (Queue/Deque Variant) | [`graphs/breadth_first_search_2.zig`](graphs/breadth_first_search_2.zig) | O(V + E) |
| Depth-First Search (Iterative Variant) | [`graphs/depth_first_search_2.zig`](graphs/depth_first_search_2.zig) | O(V + E) |
| Dijkstra (Matrix Float Variant) | [`graphs/dijkstra_2.zig`](graphs/dijkstra_2.zig) | O(V²) |
| Dijkstra (Alternate Matrix Variant) | [`graphs/dijkstra_alternate.zig`](graphs/dijkstra_alternate.zig) | O(V²) |
| Greedy Minimum Vertex Cover (Approx.) | [`graphs/greedy_min_vertex_cover.zig`](graphs/greedy_min_vertex_cover.zig) | O(V² + V·E) |
| Matching Minimum Vertex Cover (Approx.) | [`graphs/matching_min_vertex_cover.zig`](graphs/matching_min_vertex_cover.zig) | O(V³) worst |
| Karger Minimum Cut | [`graphs/karger_min_cut.zig`](graphs/karger_min_cut.zig) | O(trials·V·E) |
| Random Graph Generator | [`graphs/random_graph_generator.zig`](graphs/random_graph_generator.zig) | O(V²) |
| Markov Chain Transition Simulation | [`graphs/markov_chain.zig`](graphs/markov_chain.zig) | O(steps·N) |
| Kahn Longest Distance in DAG | [`graphs/kahn_longest_distance.zig`](graphs/kahn_longest_distance.zig) | O(V + E) |
| Graph Adjacency List Data Structure | [`graphs/graph_adjacency_list.zig`](graphs/graph_adjacency_list.zig) | O(1) avg edge insert/query, O(deg) removal |
| Graph Adjacency Matrix Data Structure | [`graphs/graph_adjacency_matrix.zig`](graphs/graph_adjacency_matrix.zig) | O(1) edge query/update, O(V²) vertex resize |
| A* Search (Python Filename Alias) | [`graphs/a_star.zig`](graphs/a_star.zig) | O(V² + E) |
| Ant Colony Optimization for TSP | [`graphs/ant_colony_optimization_algorithms.zig`](graphs/ant_colony_optimization_algorithms.zig) | O(iterations · ants · n²) |
| Basic Graph Utilities | [`graphs/basic_graphs.zig`](graphs/basic_graphs.zig) | depends on helper; O(V + E) to O(V² + E) |
| Bi-directional Dijkstra (Python Filename Alias) | [`graphs/bi_directional_dijkstra.zig`](graphs/bi_directional_dijkstra.zig) | O(V² + E) |
| Bidirectional A* Search | [`graphs/bidirectional_a_star.zig`](graphs/bidirectional_a_star.zig) | O(R · C · frontier_sort) |
| Boruvka MST (Python Filename Alias) | [`graphs/boruvka.zig`](graphs/boruvka.zig) | O(E log V) |
| Breadth-First Search (Python Filename Alias) | [`graphs/breadth_first_search.zig`](graphs/breadth_first_search.zig) | O(V + E) |
| BFS Shortest Path Variant 2 (Python Filename Alias) | [`graphs/breadth_first_search_shortest_path_2.zig`](graphs/breadth_first_search_shortest_path_2.zig) | O(V + E) |
| 0-1 BFS Shortest Path (Python Filename Alias) | [`graphs/breadth_first_search_zero_one_shortest_path.zig`](graphs/breadth_first_search_zero_one_shortest_path.zig) | O(V + E) |
| Bipartite Check (Python Filename Alias) | [`graphs/check_bipartite.zig`](graphs/check_bipartite.zig) | O(V + E) |
| Cycle Detection (Python Filename Alias) | [`graphs/check_cycle.zig`](graphs/check_cycle.zig) | O(V + E) |
| Depth-First Search (Python Filename Alias) | [`graphs/depth_first_search.zig`](graphs/depth_first_search.zig) | O(V + E) |
| Dijkstra Shortest Path (Python Filename Alias) | [`graphs/dijkstra_algorithm.zig`](graphs/dijkstra_algorithm.zig) | O(V² + E) |
| Dinic Max Flow (Python Filename Alias) | [`graphs/dinic.zig`](graphs/dinic.zig) | O(V²·E) |
| Directed and Undirected Weighted Graph Utilities | [`graphs/directed_and_undirected_weighted_graph.zig`](graphs/directed_and_undirected_weighted_graph.zig) | depends on operation; O(V + E) to O(V² + E) |
| Edmonds-Karp with Multiple Sources/Sinks | [`graphs/edmonds_karp_multiple_source_and_sink.zig`](graphs/edmonds_karp_multiple_source_and_sink.zig) | O(V⁵) worst-case |
| Eulerian Path/Circuit (Python Filename Alias) | [`graphs/eulerian_path_and_circuit_for_undirected_graph.zig`](graphs/eulerian_path_and_circuit_for_undirected_graph.zig) | O(V + E) |
| Bridges (Python Filename Alias) | [`graphs/finding_bridges.zig`](graphs/finding_bridges.zig) | O(V + E) |
| Frequent Pattern Graph Miner | [`graphs/frequent_pattern_graph_miner.zig`](graphs/frequent_pattern_graph_miner.zig) | input-dependent; worst-case exponential in DFS path enumeration |
| Graph Topological Sort (Python Filename Alias) | [`graphs/g_topological_sort.zig`](graphs/g_topological_sort.zig) | O(V + E) |
| Gale-Shapley Bigraph (Python Filename Alias) | [`graphs/gale_shapley_bigraph.zig`](graphs/gale_shapley_bigraph.zig) | O(n²) |
| Graph List Data Structure (Python Filename Alias) | [`graphs/graph_list.zig`](graphs/graph_list.zig) | O(1) avg edge insert/query, O(deg) removal |
| Floyd-Warshall (Python Filename Alias) | [`graphs/graphs_floyd_warshall.zig`](graphs/graphs_floyd_warshall.zig) | O(V³) |
| Kahn Longest Distance (Python Filename Alias) | [`graphs/kahns_algorithm_long.zig`](graphs/kahns_algorithm_long.zig) | O(V + E) |
| Kahn Topological Sort (Python Filename Alias) | [`graphs/kahns_algorithm_topo.zig`](graphs/kahns_algorithm_topo.zig) | O(V + E) |
| Karger Minimum Cut (Python Filename Alias) | [`graphs/karger.zig`](graphs/karger.zig) | O(trials·V·E) |
| Lanczos Eigenvectors | [`graphs/lanczos_eigenvectors.zig`](graphs/lanczos_eigenvectors.zig) | O(k · (V + E + k) + k³) |
| Boruvka MST Variant (Python Filename Alias) | [`graphs/minimum_spanning_tree_boruvka.zig`](graphs/minimum_spanning_tree_boruvka.zig) | O(E log V) |
| Kruskal MST Variant | [`graphs/minimum_spanning_tree_kruskal.zig`](graphs/minimum_spanning_tree_kruskal.zig) | O(E log E) |
| Kruskal MST Variant 2 | [`graphs/minimum_spanning_tree_kruskal2.zig`](graphs/minimum_spanning_tree_kruskal2.zig) | O(E log E) |
| Prim MST Variant | [`graphs/minimum_spanning_tree_prims.zig`](graphs/minimum_spanning_tree_prims.zig) | O(V² + E) |
| Prim MST Variant 2 | [`graphs/minimum_spanning_tree_prims2.zig`](graphs/minimum_spanning_tree_prims2.zig) | O(V² + E) |
| Multi-Heuristic A* | [`graphs/multi_heuristic_astar.zig`](graphs/multi_heuristic_astar.zig) | O(H · V² + H · E · V) in array-backed form |
| Kosaraju SCC (Python Filename Alias) | [`graphs/scc_kosaraju.zig`](graphs/scc_kosaraju.zig) | O(V + E) |
| Strongly Connected Components (Python Filename Alias) | [`graphs/strongly_connected_components.zig`](graphs/strongly_connected_components.zig) | O(V + E) |
| Tarjan SCC (Python Filename Alias) | [`graphs/tarjans_scc.zig`](graphs/tarjans_scc.zig) | O(V + E) |

### 图算法 (82)

| 算法 | 文件 | 复杂度 |
|------|------|--------|
| 广度优先搜索 (BFS) | [`graphs/bfs.zig`](graphs/bfs.zig) | O(V + E) |
| 深度优先搜索 (DFS) | [`graphs/dfs.zig`](graphs/dfs.zig) | O(V + E) |
| Dijkstra 最短路径 | [`graphs/dijkstra.zig`](graphs/dijkstra.zig) | O(V² + E) |
| A* 搜索 | [`graphs/a_star_search.zig`](graphs/a_star_search.zig) | O(V² + E) |
| Tarjan 强连通分量 | [`graphs/tarjan_scc.zig`](graphs/tarjan_scc.zig) | O(V + E) |
| 桥边（割边） | [`graphs/bridges.zig`](graphs/bridges.zig) | O(V + E) |
| 欧拉路径/回路（无向图） | [`graphs/eulerian_path_circuit_undirected.zig`](graphs/eulerian_path_circuit_undirected.zig) | O(V + E) |
| Ford-Fulkerson 最大流 | [`graphs/ford_fulkerson.zig`](graphs/ford_fulkerson.zig) | O(V · E²) |
| 二分图检查（BFS） | [`graphs/bipartite_check_bfs.zig`](graphs/bipartite_check_bfs.zig) | O(V + E) |
| Bellman-Ford 最短路径 | [`graphs/bellman_ford.zig`](graphs/bellman_ford.zig) | O(V·E) |
| 拓扑排序 | [`graphs/topological_sort.zig`](graphs/topological_sort.zig) | O(V + E) |
| Floyd-Warshall | [`graphs/floyd_warshall.zig`](graphs/floyd_warshall.zig) | O(V³) |
| 有向图环检测 | [`graphs/detect_cycle.zig`](graphs/detect_cycle.zig) | O(V + E) |
| 连通分量计数 | [`graphs/connected_components.zig`](graphs/connected_components.zig) | O(V + E) |
| Kruskal 最小生成树 | [`graphs/kruskal.zig`](graphs/kruskal.zig) | O(E log E) |
| Prim 最小生成树 | [`graphs/prim.zig`](graphs/prim.zig) | O(V² + E) |
| 割点检测 | [`graphs/articulation_points.zig`](graphs/articulation_points.zig) | O(V + E) |
| Kosaraju 强连通分量 | [`graphs/kosaraju_scc.zig`](graphs/kosaraju_scc.zig) | O(V + E) |
| Kahn 拓扑排序 | [`graphs/kahn_topological_sort.zig`](graphs/kahn_topological_sort.zig) | O(V + E) |
| BFS 最短路径 | [`graphs/breadth_first_search_shortest_path.zig`](graphs/breadth_first_search_shortest_path.zig) | O(V + E) |
| Boruvka 最小生成树 | [`graphs/boruvka_mst.zig`](graphs/boruvka_mst.zig) | O(E log V) |
| 0-1 BFS 最短路径 | [`graphs/zero_one_bfs_shortest_path.zig`](graphs/zero_one_bfs_shortest_path.zig) | O(V + E) |
| 双向 BFS 路径搜索 | [`graphs/bidirectional_breadth_first_search.zig`](graphs/bidirectional_breadth_first_search.zig) | O(V + E) |
| 二值网格 Dijkstra | [`graphs/dijkstra_binary_grid.zig`](graphs/dijkstra_binary_grid.zig) | O((R·C)²) |
| Even Tree（偶数森林最大切边） | [`graphs/even_tree.zig`](graphs/even_tree.zig) | O(V + E) |
| Gale-Shapley 稳定匹配 | [`graphs/gale_shapley_stable_matching.zig`](graphs/gale_shapley_stable_matching.zig) | O(n²) |
| PageRank（迭代版） | [`graphs/page_rank.zig`](graphs/page_rank.zig) | O(iterations·(V + E)) |
| 双向 Dijkstra | [`graphs/bidirectional_dijkstra.zig`](graphs/bidirectional_dijkstra.zig) | O(V² + E) |
| 贪心最佳优先搜索 | [`graphs/greedy_best_first.zig`](graphs/greedy_best_first.zig) | O(V²) |
| Dinic 最大流 | [`graphs/dinic_max_flow.zig`](graphs/dinic_max_flow.zig) | O(V²·E) |
| 双向搜索 | [`graphs/bidirectional_search.zig`](graphs/bidirectional_search.zig) | O(V + E) |
| 最小路径和 | [`graphs/minimum_path_sum.zig`](graphs/minimum_path_sum.zig) | O(rows·cols) |
| 图深拷贝 | [`graphs/deep_clone_graph.zig`](graphs/deep_clone_graph.zig) | O(V + E) |
| Dijkstra（邻接矩阵） | [`graphs/dijkstra_matrix.zig`](graphs/dijkstra_matrix.zig) | O(V²) |
| 广度优先搜索（队列/双端队列变体） | [`graphs/breadth_first_search_2.zig`](graphs/breadth_first_search_2.zig) | O(V + E) |
| 深度优先搜索（迭代变体） | [`graphs/depth_first_search_2.zig`](graphs/depth_first_search_2.zig) | O(V + E) |
| Dijkstra（浮点邻接矩阵变体） | [`graphs/dijkstra_2.zig`](graphs/dijkstra_2.zig) | O(V²) |
| Dijkstra（另一邻接矩阵变体） | [`graphs/dijkstra_alternate.zig`](graphs/dijkstra_alternate.zig) | O(V²) |
| 贪心最小顶点覆盖（近似） | [`graphs/greedy_min_vertex_cover.zig`](graphs/greedy_min_vertex_cover.zig) | O(V² + V·E) |
| 匹配最小顶点覆盖（近似） | [`graphs/matching_min_vertex_cover.zig`](graphs/matching_min_vertex_cover.zig) | O(V³) worst |
| Karger 最小割 | [`graphs/karger_min_cut.zig`](graphs/karger_min_cut.zig) | O(trials·V·E) |
| 随机图生成器 | [`graphs/random_graph_generator.zig`](graphs/random_graph_generator.zig) | O(V²) |
| 马尔可夫链转移模拟 | [`graphs/markov_chain.zig`](graphs/markov_chain.zig) | O(steps·N) |
| Kahn DAG 最长距离 | [`graphs/kahn_longest_distance.zig`](graphs/kahn_longest_distance.zig) | O(V + E) |
| 图邻接表数据结构 | [`graphs/graph_adjacency_list.zig`](graphs/graph_adjacency_list.zig) | O(1) avg edge insert/query, O(deg) removal |
| 图邻接矩阵数据结构 | [`graphs/graph_adjacency_matrix.zig`](graphs/graph_adjacency_matrix.zig) | O(1) edge query/update, O(V²) vertex resize |
| A* 搜索（Python 文件名兼容层） | [`graphs/a_star.zig`](graphs/a_star.zig) | O(V² + E) |
| 蚁群优化旅行商问题 | [`graphs/ant_colony_optimization_algorithms.zig`](graphs/ant_colony_optimization_algorithms.zig) | O(iterations · ants · n²) |
| 基础图算法工具集 | [`graphs/basic_graphs.zig`](graphs/basic_graphs.zig) | 随辅助函数而变；O(V + E) 到 O(V² + E) |
| 双向 Dijkstra（Python 文件名兼容层） | [`graphs/bi_directional_dijkstra.zig`](graphs/bi_directional_dijkstra.zig) | O(V² + E) |
| 双向 A* 搜索 | [`graphs/bidirectional_a_star.zig`](graphs/bidirectional_a_star.zig) | O(R · C · frontier_sort) |
| Boruvka 最小生成树（Python 文件名兼容层） | [`graphs/boruvka.zig`](graphs/boruvka.zig) | O(E log V) |
| 广度优先搜索（Python 文件名兼容层） | [`graphs/breadth_first_search.zig`](graphs/breadth_first_search.zig) | O(V + E) |
| BFS 最短路径变体 2（Python 文件名兼容层） | [`graphs/breadth_first_search_shortest_path_2.zig`](graphs/breadth_first_search_shortest_path_2.zig) | O(V + E) |
| 0-1 BFS 最短路径（Python 文件名兼容层） | [`graphs/breadth_first_search_zero_one_shortest_path.zig`](graphs/breadth_first_search_zero_one_shortest_path.zig) | O(V + E) |
| 二分图检查（Python 文件名兼容层） | [`graphs/check_bipartite.zig`](graphs/check_bipartite.zig) | O(V + E) |
| 环检测（Python 文件名兼容层） | [`graphs/check_cycle.zig`](graphs/check_cycle.zig) | O(V + E) |
| 深度优先搜索（Python 文件名兼容层） | [`graphs/depth_first_search.zig`](graphs/depth_first_search.zig) | O(V + E) |
| Dijkstra 最短路径（Python 文件名兼容层） | [`graphs/dijkstra_algorithm.zig`](graphs/dijkstra_algorithm.zig) | O(V² + E) |
| Dinic 最大流（Python 文件名兼容层） | [`graphs/dinic.zig`](graphs/dinic.zig) | O(V²·E) |
| 有向/无向加权图工具集 | [`graphs/directed_and_undirected_weighted_graph.zig`](graphs/directed_and_undirected_weighted_graph.zig) | 随操作而变；O(V + E) 到 O(V² + E) |
| 多源多汇 Edmonds-Karp 最大流 | [`graphs/edmonds_karp_multiple_source_and_sink.zig`](graphs/edmonds_karp_multiple_source_and_sink.zig) | 最坏 O(V⁵) |
| 欧拉路径/回路（Python 文件名兼容层） | [`graphs/eulerian_path_and_circuit_for_undirected_graph.zig`](graphs/eulerian_path_and_circuit_for_undirected_graph.zig) | O(V + E) |
| 桥边检测（Python 文件名兼容层） | [`graphs/finding_bridges.zig`](graphs/finding_bridges.zig) | O(V + E) |
| 频繁模式图挖掘 | [`graphs/frequent_pattern_graph_miner.zig`](graphs/frequent_pattern_graph_miner.zig) | 依输入而定；最坏为 DFS 路径枚举指数级 |
| 图拓扑排序（Python 文件名兼容层） | [`graphs/g_topological_sort.zig`](graphs/g_topological_sort.zig) | O(V + E) |
| Gale-Shapley 二分图稳定匹配（Python 文件名兼容层） | [`graphs/gale_shapley_bigraph.zig`](graphs/gale_shapley_bigraph.zig) | O(n²) |
| 图列表数据结构（Python 文件名兼容层） | [`graphs/graph_list.zig`](graphs/graph_list.zig) | O(1) avg edge insert/query, O(deg) removal |
| Floyd-Warshall（Python 文件名兼容层） | [`graphs/graphs_floyd_warshall.zig`](graphs/graphs_floyd_warshall.zig) | O(V³) |
| Kahn 最长距离（Python 文件名兼容层） | [`graphs/kahns_algorithm_long.zig`](graphs/kahns_algorithm_long.zig) | O(V + E) |
| Kahn 拓扑排序（Python 文件名兼容层） | [`graphs/kahns_algorithm_topo.zig`](graphs/kahns_algorithm_topo.zig) | O(V + E) |
| Karger 最小割（Python 文件名兼容层） | [`graphs/karger.zig`](graphs/karger.zig) | O(trials·V·E) |
| Lanczos 特征向量 | [`graphs/lanczos_eigenvectors.zig`](graphs/lanczos_eigenvectors.zig) | O(k · (V + E + k) + k³) |
| Boruvka 最小生成树变体 | [`graphs/minimum_spanning_tree_boruvka.zig`](graphs/minimum_spanning_tree_boruvka.zig) | O(E log V) |
| Kruskal 最小生成树变体 | [`graphs/minimum_spanning_tree_kruskal.zig`](graphs/minimum_spanning_tree_kruskal.zig) | O(E log E) |
| Kruskal 最小生成树变体 2 | [`graphs/minimum_spanning_tree_kruskal2.zig`](graphs/minimum_spanning_tree_kruskal2.zig) | O(E log E) |
| Prim 最小生成树变体 | [`graphs/minimum_spanning_tree_prims.zig`](graphs/minimum_spanning_tree_prims.zig) | O(V² + E) |
| Prim 最小生成树变体 2 | [`graphs/minimum_spanning_tree_prims2.zig`](graphs/minimum_spanning_tree_prims2.zig) | O(V² + E) |
| 多启发式 A* | [`graphs/multi_heuristic_astar.zig`](graphs/multi_heuristic_astar.zig) | 数组后端实现下 O(H · V² + H · E · V) |
| Kosaraju 强连通分量（Python 文件名兼容层） | [`graphs/scc_kosaraju.zig`](graphs/scc_kosaraju.zig) | O(V + E) |
| 强连通分量（Python 文件名兼容层） | [`graphs/strongly_connected_components.zig`](graphs/strongly_connected_components.zig) | O(V + E) |
| Tarjan 强连通分量（Python 文件名兼容层） | [`graphs/tarjans_scc.zig`](graphs/tarjans_scc.zig) | O(V + E) |

### Greedy Methods (8)

| Algorithm | File | Complexity |
|-----------|------|-----------|
| Best Time to Buy/Sell Stock | [`greedy_methods/best_time_to_buy_sell_stock.zig`](greedy_methods/best_time_to_buy_sell_stock.zig) | O(n) |
| Minimum Coin Change (Greedy) | [`greedy_methods/minimum_coin_change.zig`](greedy_methods/minimum_coin_change.zig) | O(n·k) |
| Minimum Waiting Time | [`greedy_methods/minimum_waiting_time.zig`](greedy_methods/minimum_waiting_time.zig) | O(n log n) |
| Fractional Knapsack | [`greedy_methods/fractional_knapsack.zig`](greedy_methods/fractional_knapsack.zig) | O(n log n) |
| Activity Selection | [`greedy_methods/activity_selection.zig`](greedy_methods/activity_selection.zig) | O(n) |
| Huffman Coding | [`greedy_methods/huffman_coding.zig`](greedy_methods/huffman_coding.zig) | O(n + σ log σ) |
| Job Sequencing with Deadlines | [`greedy_methods/job_sequencing_with_deadline.zig`](greedy_methods/job_sequencing_with_deadline.zig) | O(n log n + n·d) |
| Gas Station | [`greedy_methods/gas_station.zig`](greedy_methods/gas_station.zig) | O(n) |

### 贪心算法 (8)

| 算法 | 文件 | 复杂度 |
|------|------|--------|
| 买卖股票最佳时机 | [`greedy_methods/best_time_to_buy_sell_stock.zig`](greedy_methods/best_time_to_buy_sell_stock.zig) | O(n) |
| 最少硬币数（贪心） | [`greedy_methods/minimum_coin_change.zig`](greedy_methods/minimum_coin_change.zig) | O(n·k) |
| 最小等待时间 | [`greedy_methods/minimum_waiting_time.zig`](greedy_methods/minimum_waiting_time.zig) | O(n log n) |
| 分数背包 | [`greedy_methods/fractional_knapsack.zig`](greedy_methods/fractional_knapsack.zig) | O(n log n) |
| 活动选择 | [`greedy_methods/activity_selection.zig`](greedy_methods/activity_selection.zig) | O(n) |
| 哈夫曼编码 | [`greedy_methods/huffman_coding.zig`](greedy_methods/huffman_coding.zig) | O(n + σ log σ) |
| 截止时间作业调度 | [`greedy_methods/job_sequencing_with_deadline.zig`](greedy_methods/job_sequencing_with_deadline.zig) | O(n log n + n·d) |
| 加油站环路 | [`greedy_methods/gas_station.zig`](greedy_methods/gas_station.zig) | O(n) |

### Matrix (20)

| Algorithm | File | Complexity |
|-----------|------|-----------|
| Matrix Multiply | [`matrix/matrix_multiply.zig`](matrix/matrix_multiply.zig) | O(m·k·n) |
| Matrix Transpose | [`matrix/matrix_transpose.zig`](matrix/matrix_transpose.zig) | O(m·n) |
| Rotate Matrix 90° | [`matrix/rotate_matrix.zig`](matrix/rotate_matrix.zig) | O(n²) |
| Spiral Print | [`matrix/spiral_print.zig`](matrix/spiral_print.zig) | O(m·n) |
| Pascal's Triangle | [`matrix/pascal_triangle.zig`](matrix/pascal_triangle.zig) | O(n²) |
| Binary Search Matrix | [`matrix/binary_search_matrix.zig`](matrix/binary_search_matrix.zig) | O(r log c) |
| Count Negative Numbers in Sorted Matrix | [`matrix/count_negative_numbers_in_sorted_matrix.zig`](matrix/count_negative_numbers_in_sorted_matrix.zig) | O(r log c) |
| Count Paths | [`matrix/count_paths.zig`](matrix/count_paths.zig) | worst-case exponential |
| Count Islands in Matrix | [`matrix/count_islands_in_matrix.zig`](matrix/count_islands_in_matrix.zig) | O(r·c) |
| Cramer's Rule (2x2) | [`matrix/cramers_rule_2x2.zig`](matrix/cramers_rule_2x2.zig) | O(1) |
| Largest Square Area in Matrix | [`matrix/largest_square_area_in_matrix.zig`](matrix/largest_square_area_in_matrix.zig) | O(r·c) |
| Max Area of Island | [`matrix/max_area_of_island.zig`](matrix/max_area_of_island.zig) | O(r·c) |
| Median Matrix | [`matrix/median_matrix.zig`](matrix/median_matrix.zig) | O(n log n) |
| Searching in Sorted Matrix | [`matrix/searching_in_sorted_matrix.zig`](matrix/searching_in_sorted_matrix.zig) | O(r + c) |
| Validate Sudoku Board | [`matrix/validate_sudoku_board.zig`](matrix/validate_sudoku_board.zig) | O(1) |
| Matrix Equalization | [`matrix/matrix_equalization.zig`](matrix/matrix_equalization.zig) | O(u · n) |
| Nth Fibonacci Using Matrix Exponentiation | [`matrix/nth_fibonacci_using_matrix_exponentiation.zig`](matrix/nth_fibonacci_using_matrix_exponentiation.zig) | O(log n) |
| Matrix Operation Utilities | [`matrix/matrix_operation.zig`](matrix/matrix_operation.zig) | varies by operation; determinant/inverse O(n!) |
| Inverse Of Matrix (2x2 / 3x3 Reference Variant) | [`matrix/inverse_of_matrix.zig`](matrix/inverse_of_matrix.zig) | O(1) |
| Matrix Multiplication (Recursive) | [`matrix/matrix_multiplication_recursion.zig`](matrix/matrix_multiplication_recursion.zig) | O(n³) |

### 矩阵 (20)

| 算法 | 文件 | 复杂度 |
|------|------|--------|
| 矩阵乘法 | [`matrix/matrix_multiply.zig`](matrix/matrix_multiply.zig) | O(m·k·n) |
| 矩阵转置 | [`matrix/matrix_transpose.zig`](matrix/matrix_transpose.zig) | O(m·n) |
| 矩阵旋转 90° | [`matrix/rotate_matrix.zig`](matrix/rotate_matrix.zig) | O(n²) |
| 螺旋打印 | [`matrix/spiral_print.zig`](matrix/spiral_print.zig) | O(m·n) |
| 杨辉三角 | [`matrix/pascal_triangle.zig`](matrix/pascal_triangle.zig) | O(n²) |
| 矩阵二分查找 | [`matrix/binary_search_matrix.zig`](matrix/binary_search_matrix.zig) | O(r log c) |
| 递减排序矩阵中的负数计数 | [`matrix/count_negative_numbers_in_sorted_matrix.zig`](matrix/count_negative_numbers_in_sorted_matrix.zig) | O(r log c) |
| 网格路径计数 | [`matrix/count_paths.zig`](matrix/count_paths.zig) | 最坏指数级 |
| 矩阵中的岛屿计数 | [`matrix/count_islands_in_matrix.zig`](matrix/count_islands_in_matrix.zig) | O(r·c) |
| 克拉默法则（2x2） | [`matrix/cramers_rule_2x2.zig`](matrix/cramers_rule_2x2.zig) | O(1) |
| 最大全 1 正方形边长 | [`matrix/largest_square_area_in_matrix.zig`](matrix/largest_square_area_in_matrix.zig) | O(r·c) |
| 岛屿最大面积 | [`matrix/max_area_of_island.zig`](matrix/max_area_of_island.zig) | O(r·c) |
| 矩阵中位数 | [`matrix/median_matrix.zig`](matrix/median_matrix.zig) | O(n log n) |
| 有序矩阵查找 | [`matrix/searching_in_sorted_matrix.zig`](matrix/searching_in_sorted_matrix.zig) | O(r + c) |
| 数独棋盘有效性校验 | [`matrix/validate_sudoku_board.zig`](matrix/validate_sudoku_board.zig) | O(1) |
| 矩阵均衡化 | [`matrix/matrix_equalization.zig`](matrix/matrix_equalization.zig) | O(u · n) |
| 矩阵快速幂求第 n 个 Fibonacci | [`matrix/nth_fibonacci_using_matrix_exponentiation.zig`](matrix/nth_fibonacci_using_matrix_exponentiation.zig) | O(log n) |
| 矩阵操作工具 | [`matrix/matrix_operation.zig`](matrix/matrix_operation.zig) | 随操作而变；det/inverse 最坏 O(n!) |
| 矩阵求逆（2x2 / 3x3 参考实现语义） | [`matrix/inverse_of_matrix.zig`](matrix/inverse_of_matrix.zig) | O(1) |
| 递归矩阵乘法 | [`matrix/matrix_multiplication_recursion.zig`](matrix/matrix_multiplication_recursion.zig) | O(n³) |

### Backtracking (21)

| Algorithm | File | Complexity |
|-----------|------|-----------|
| Permutations | [`backtracking/permutations.zig`](backtracking/permutations.zig) | O(n! · n) |
| Combinations | [`backtracking/combinations.zig`](backtracking/combinations.zig) | O(C(n,k)) |
| Subsets | [`backtracking/subsets.zig`](backtracking/subsets.zig) | O(2ⁿ) |
| Generate Parentheses | [`backtracking/generate_parentheses.zig`](backtracking/generate_parentheses.zig) | O(Catalan(n)) |
| Generate Parentheses (Iterative) | [`backtracking/generate_parentheses_iterative.zig`](backtracking/generate_parentheses_iterative.zig) | O(2^(2n)) |
| N-Queens | [`backtracking/n_queens.zig`](backtracking/n_queens.zig) | O(n!) |
| Sudoku Solver | [`backtracking/sudoku_solver.zig`](backtracking/sudoku_solver.zig) | O(9^m) |
| Word Search | [`backtracking/word_search.zig`](backtracking/word_search.zig) | O(rows · cols · 4^L) |
| Rat in a Maze | [`backtracking/rat_in_maze.zig`](backtracking/rat_in_maze.zig) | worst-case exponential |
| Combination Sum | [`backtracking/combination_sum.zig`](backtracking/combination_sum.zig) | worst-case exponential |
| Power Sum | [`backtracking/power_sum.zig`](backtracking/power_sum.zig) | worst-case exponential |
| Word Break (Backtracking) | [`backtracking/word_break.zig`](backtracking/word_break.zig) | worst-case exponential |
| Sum of Subsets | [`backtracking/sum_of_subsets.zig`](backtracking/sum_of_subsets.zig) | worst-case exponential |
| Hamiltonian Cycle | [`backtracking/hamiltonian_cycle.zig`](backtracking/hamiltonian_cycle.zig) | worst-case exponential |
| All Subsequences | [`backtracking/all_subsequences.zig`](backtracking/all_subsequences.zig) | O(2ⁿ) |
| Match Word Pattern | [`backtracking/match_word_pattern.zig`](backtracking/match_word_pattern.zig) | worst-case exponential |
| Minimax | [`backtracking/minimax.zig`](backtracking/minimax.zig) | O(2^h) |
| Graph Coloring (M-Coloring) | [`backtracking/coloring.zig`](backtracking/coloring.zig) | O(m^n) |
| Knight Tour | [`backtracking/knight_tour.zig`](backtracking/knight_tour.zig) | worst-case exponential |
| Word Ladder (Backtracking) | [`backtracking/word_ladder.zig`](backtracking/word_ladder.zig) | worst-case exponential |
| N-Queens (Math DFS) | [`backtracking/n_queens_math.zig`](backtracking/n_queens_math.zig) | O(n!) |

### 回溯算法 (21)

| 算法 | 文件 | 复杂度 |
|------|------|--------|
| 全排列 | [`backtracking/permutations.zig`](backtracking/permutations.zig) | O(n! · n) |
| 组合 | [`backtracking/combinations.zig`](backtracking/combinations.zig) | O(C(n,k)) |
| 子集（幂集） | [`backtracking/subsets.zig`](backtracking/subsets.zig) | O(2ⁿ) |
| 生成括号 | [`backtracking/generate_parentheses.zig`](backtracking/generate_parentheses.zig) | O(Catalan(n)) |
| 生成括号（迭代） | [`backtracking/generate_parentheses_iterative.zig`](backtracking/generate_parentheses_iterative.zig) | O(2^(2n)) |
| N 皇后 | [`backtracking/n_queens.zig`](backtracking/n_queens.zig) | O(n!) |
| 数独求解 | [`backtracking/sudoku_solver.zig`](backtracking/sudoku_solver.zig) | O(9^m) |
| 单词搜索 | [`backtracking/word_search.zig`](backtracking/word_search.zig) | O(rows · cols · 4^L) |
| 迷宫老鼠问题 | [`backtracking/rat_in_maze.zig`](backtracking/rat_in_maze.zig) | 最坏指数级 |
| 组合总和 | [`backtracking/combination_sum.zig`](backtracking/combination_sum.zig) | 最坏指数级 |
| 幂和问题 | [`backtracking/power_sum.zig`](backtracking/power_sum.zig) | 最坏指数级 |
| 单词拆分（回溯） | [`backtracking/word_break.zig`](backtracking/word_break.zig) | 最坏指数级 |
| 子集和问题 | [`backtracking/sum_of_subsets.zig`](backtracking/sum_of_subsets.zig) | 最坏指数级 |
| 哈密顿回路 | [`backtracking/hamiltonian_cycle.zig`](backtracking/hamiltonian_cycle.zig) | 最坏指数级 |
| 全部子序列 | [`backtracking/all_subsequences.zig`](backtracking/all_subsequences.zig) | O(2ⁿ) |
| 单词模式匹配（回溯） | [`backtracking/match_word_pattern.zig`](backtracking/match_word_pattern.zig) | 最坏指数级 |
| 极小化极大算法 | [`backtracking/minimax.zig`](backtracking/minimax.zig) | O(2^h) |
| 图着色（M 着色） | [`backtracking/coloring.zig`](backtracking/coloring.zig) | O(m^n) |
| 骑士巡游 | [`backtracking/knight_tour.zig`](backtracking/knight_tour.zig) | 最坏指数级 |
| 单词阶梯（回溯） | [`backtracking/word_ladder.zig`](backtracking/word_ladder.zig) | 最坏指数级 |
| N 皇后（数学 DFS） | [`backtracking/n_queens_math.zig`](backtracking/n_queens_math.zig) | O(n!) |

### Knapsack (3)

| Algorithm | File | Complexity |
|-----------|------|-----------|
| 0/1 Knapsack (Memoized Recursive) | [`knapsack/knapsack.zig`](knapsack/knapsack.zig) | O(n × W) |
| Recursive Approach Knapsack | [`knapsack/recursive_approach_knapsack.zig`](knapsack/recursive_approach_knapsack.zig) | O(2ⁿ) |
| Greedy Knapsack | [`knapsack/greedy_knapsack.zig`](knapsack/greedy_knapsack.zig) | O(n log n) |

### 背包 (3)

| 算法 | 文件 | 复杂度 |
|------|------|--------|
| 0/1 背包（记忆化递归） | [`knapsack/knapsack.zig`](knapsack/knapsack.zig) | O(n × W) |
| 递归背包 | [`knapsack/recursive_approach_knapsack.zig`](knapsack/recursive_approach_knapsack.zig) | O(2ⁿ) |
| 贪心背包 | [`knapsack/greedy_knapsack.zig`](knapsack/greedy_knapsack.zig) | O(n log n) |
