//! Binary Tree Node Sum - Zig implementation
//! Reference: https://github.com/TheAlgorithms/Python/blob/master/data_structures/binary_tree/binary_tree_node_sum.py

const std = @import("std");
const testing = std.testing;

pub const Node = struct {
    value: i64,
    left: ?*Node = null,
    right: ?*Node = null,
};

/// Computes sum of node values using depth-first traversal.
/// Time complexity: O(n), Space complexity: O(h)
pub fn depthFirstSearch(node: ?*const Node) i64 {
    const n = node orelse return 0;
    return n.value + depthFirstSearch(n.left) + depthFirstSearch(n.right);
}

/// Returns the sum of all nodes in the tree.
/// Time complexity: O(n), Space complexity: O(h)
pub fn binaryTreeNodeSum(root: ?*const Node) i64 {
    return depthFirstSearch(root);
}

test "binary tree node sum: python doctest sequence" {
    var tree = Node{ .value = 10 };
    try testing.expectEqual(@as(i64, 10), binaryTreeNodeSum(&tree));

    var n5 = Node{ .value = 5 };
    tree.left = &n5;
    try testing.expectEqual(@as(i64, 15), binaryTreeNodeSum(&tree));

    var n_3 = Node{ .value = -3 };
    tree.right = &n_3;
    try testing.expectEqual(@as(i64, 12), binaryTreeNodeSum(&tree));

    var n12 = Node{ .value = 12 };
    n5.left = &n12;
    try testing.expectEqual(@as(i64, 24), binaryTreeNodeSum(&tree));

    var n8 = Node{ .value = 8 };
    var n0 = Node{ .value = 0 };
    n_3.left = &n8;
    n_3.right = &n0;
    try testing.expectEqual(@as(i64, 32), binaryTreeNodeSum(&tree));
}

test "binary tree node sum: boundary" {
    try testing.expectEqual(@as(i64, 0), binaryTreeNodeSum(null));

    var leaf = Node{ .value = -42 };
    try testing.expectEqual(@as(i64, -42), binaryTreeNodeSum(&leaf));
}

test "binary tree node sum: extreme complete tree" {
    const n: usize = 100_000;
    const nodes = try testing.allocator.alloc(Node, n);
    defer testing.allocator.free(nodes);

    var expected: i64 = 0;
    for (0..n) |i| {
        const value: i64 = @as(i64, @intCast(i % 17)) - 8;
        expected += value;
        nodes[i] = .{ .value = value };
    }

    for (0..n) |i| {
        const left_idx = i * 2 + 1;
        const right_idx = i * 2 + 2;
        nodes[i].left = if (left_idx < n) &nodes[left_idx] else null;
        nodes[i].right = if (right_idx < n) &nodes[right_idx] else null;
    }

    try testing.expectEqual(expected, binaryTreeNodeSum(&nodes[0]));
}
