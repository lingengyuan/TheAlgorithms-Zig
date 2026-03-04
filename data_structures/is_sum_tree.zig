//! Is Sum Tree - Zig implementation
//! Reference: https://github.com/TheAlgorithms/Python/blob/master/data_structures/binary_tree/is_sum_tree.py

const std = @import("std");
const testing = std.testing;

pub const Node = struct {
    data: i64,
    left: ?*Node = null,
    right: ?*Node = null,
};

const SumInfo = struct {
    is_sum_node: bool,
    sum: i64,
};

fn sumInfo(node: ?*const Node) SumInfo {
    const n = node orelse return .{ .is_sum_node = true, .sum = 0 };

    if (n.left == null and n.right == null) {
        return .{ .is_sum_node = true, .sum = n.data };
    }

    const left = sumInfo(n.left);
    const right = sumInfo(n.right);

    return .{
        .is_sum_node = left.is_sum_node and right.is_sum_node and (n.data == left.sum + right.sum),
        .sum = n.data + left.sum + right.sum,
    };
}

/// Returns true if every non-leaf node equals sum(left-subtree) + sum(right-subtree).
/// Time complexity: O(n), Space complexity: O(h)
pub fn isSumTree(root: ?*const Node) bool {
    return sumInfo(root).is_sum_node;
}

fn inorderTraversal(allocator: std.mem.Allocator, root: ?*const Node) ![]i64 {
    var out = std.ArrayListUnmanaged(i64){};
    errdefer out.deinit(allocator);

    var stack = std.ArrayListUnmanaged(*const Node){};
    defer stack.deinit(allocator);

    var current = root;
    while (current != null or stack.items.len > 0) {
        while (current) |node| {
            try stack.append(allocator, node);
            current = node.left;
        }

        const node = stack.pop().?;
        try out.append(allocator, node.data);
        current = node.right;
    }

    return out.toOwnedSlice(allocator);
}

fn treeLen(allocator: std.mem.Allocator, root: ?*const Node) !usize {
    const start = root orelse return 0;

    var stack = std.ArrayListUnmanaged(*const Node){};
    defer stack.deinit(allocator);
    try stack.append(allocator, start);

    var count: usize = 0;
    while (stack.items.len > 0) {
        const node = stack.pop().?;
        count += 1;
        if (node.left) |left| try stack.append(allocator, left);
        if (node.right) |right| try stack.append(allocator, right);
    }

    return count;
}

test "is sum tree: python basic node checks" {
    var root = Node{ .data = 3 };
    try testing.expect(isSumTree(&root));

    var left = Node{ .data = 1 };
    root.left = &left;
    try testing.expect(!isSumTree(&root));

    var right = Node{ .data = 2 };
    root.right = &right;
    try testing.expect(isSumTree(&root));
}

test "is sum tree: python build_a_tree and build_a_sum_tree" {
    // build_a_tree sample (not a sum tree)
    var n11 = Node{ .data = 11 };
    var n2 = Node{ .data = 2 };
    var n29 = Node{ .data = 29 };
    var n1 = Node{ .data = 1 };
    var n7 = Node{ .data = 7 };
    var n15 = Node{ .data = 15 };
    var n40 = Node{ .data = 40 };
    var n35 = Node{ .data = 35 };

    n11.left = &n2;
    n11.right = &n29;
    n2.left = &n1;
    n2.right = &n7;
    n29.left = &n15;
    n29.right = &n40;
    n40.left = &n35;

    const in1 = try inorderTraversal(testing.allocator, &n11);
    defer testing.allocator.free(in1);
    try testing.expectEqualSlices(i64, &[_]i64{ 1, 2, 7, 11, 15, 29, 35, 40 }, in1);
    try testing.expectEqual(@as(usize, 8), try treeLen(testing.allocator, &n11));
    try testing.expect(!isSumTree(&n11));

    // build_a_sum_tree sample
    var n26 = Node{ .data = 26 };
    var n10 = Node{ .data = 10 };
    var n3 = Node{ .data = 3 };
    var n4 = Node{ .data = 4 };
    var n6 = Node{ .data = 6 };
    var n3r = Node{ .data = 3 };

    n26.left = &n10;
    n26.right = &n3;
    n10.left = &n4;
    n10.right = &n6;
    n3.right = &n3r;

    const in2 = try inorderTraversal(testing.allocator, &n26);
    defer testing.allocator.free(in2);
    try testing.expectEqualSlices(i64, &[_]i64{ 4, 10, 6, 26, 3, 3 }, in2);
    try testing.expect(isSumTree(&n26));
}

test "is sum tree: extreme perfect tree" {
    const depth: usize = 14;
    const n: usize = (@as(usize, 1) << depth) - 1;

    const nodes = try testing.allocator.alloc(Node, n);
    defer testing.allocator.free(nodes);
    const subtree_sums = try testing.allocator.alloc(i64, n);
    defer testing.allocator.free(subtree_sums);

    for (0..n) |i| {
        nodes[i] = .{ .data = 0 };
    }

    for (0..n) |i| {
        const left_idx = i * 2 + 1;
        const right_idx = i * 2 + 2;
        nodes[i].left = if (left_idx < n) &nodes[left_idx] else null;
        nodes[i].right = if (right_idx < n) &nodes[right_idx] else null;
    }

    const first_leaf = n / 2;
    for (first_leaf..n) |i| {
        nodes[i].data = 1;
        subtree_sums[i] = 1;
    }

    var i = n / 2;
    while (i > 0) {
        i -= 1;
        const left_idx = i * 2 + 1;
        const right_idx = i * 2 + 2;
        nodes[i].data = subtree_sums[left_idx] + subtree_sums[right_idx];
        subtree_sums[i] = nodes[i].data + subtree_sums[left_idx] + subtree_sums[right_idx];
    }

    try testing.expect(isSumTree(&nodes[0]));

    nodes[n - 1].data += 1;
    try testing.expect(!isSumTree(&nodes[0]));
}
