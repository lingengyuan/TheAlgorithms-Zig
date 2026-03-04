//! Binary Tree Path Sum - Zig implementation
//! Reference: https://github.com/TheAlgorithms/Python/blob/master/data_structures/binary_tree/binary_tree_path_sum.py

const std = @import("std");
const testing = std.testing;

pub const Node = struct {
    value: i64,
    left: ?*Node = null,
    right: ?*Node = null,
};

fn countFrom(node: ?*const Node, target: i64, running_sum: i64) usize {
    const n = node orelse return 0;

    const current_sum = running_sum + n.value;
    var paths: usize = 0;
    if (current_sum == target) paths += 1;

    paths += countFrom(n.left, target, current_sum);
    paths += countFrom(n.right, target, current_sum);
    return paths;
}

/// Counts downward paths whose node-value sum equals `target`.
/// Equivalent behavior to Python `BinaryTreePathSum.path_sum`.
/// Time complexity: O(n^2) worst-case, Space complexity: O(h)
pub fn pathSum(root: ?*const Node, target: i64) usize {
    const node = root orelse return 0;

    return countFrom(node, target, 0) + pathSum(node.left, target) + pathSum(node.right, target);
}

test "binary tree path sum: python primary doctest" {
    var tree = Node{ .value = 10 };
    var n5 = Node{ .value = 5 };
    var n_3 = Node{ .value = -3 };
    var n3 = Node{ .value = 3 };
    var n2 = Node{ .value = 2 };
    var n11 = Node{ .value = 11 };
    var n3_l = Node{ .value = 3 };
    var n_2 = Node{ .value = -2 };
    var n1 = Node{ .value = 1 };

    tree.left = &n5;
    tree.right = &n_3;
    n5.left = &n3;
    n5.right = &n2;
    n_3.right = &n11;
    n3.left = &n3_l;
    n3.right = &n_2;
    n2.right = &n1;

    try testing.expectEqual(@as(usize, 3), pathSum(&tree, 8));
    try testing.expectEqual(@as(usize, 2), pathSum(&tree, 7));

    var n10 = Node{ .value = 10 };
    n_3.right = &n10;
    try testing.expectEqual(@as(usize, 2), pathSum(&tree, 8));

    try testing.expectEqual(@as(usize, 0), pathSum(null, 0));
    try testing.expectEqual(@as(usize, 0), pathSum(&tree, 0));
}

test "binary tree path sum: python second doctest" {
    var tree = Node{ .value = 0 };
    var left = Node{ .value = 5 };
    var right = Node{ .value = 5 };

    tree.left = &left;
    tree.right = &right;

    try testing.expectEqual(@as(usize, 4), pathSum(&tree, 5));
    try testing.expectEqual(@as(usize, 0), pathSum(&tree, -1));
    try testing.expectEqual(@as(usize, 1), pathSum(&tree, 0));
}

test "binary tree path sum: extreme long chain" {
    const n: usize = 3_000;
    const nodes = try testing.allocator.alloc(Node, n);
    defer testing.allocator.free(nodes);

    for (0..n) |i| {
        nodes[i] = .{ .value = 1 };
    }
    for (0..n) |i| {
        nodes[i].right = if (i + 1 < n) &nodes[i + 1] else null;
    }

    try testing.expectEqual(@as(usize, n - 2), pathSum(&nodes[0], 3));
    try testing.expectEqual(@as(usize, n), pathSum(&nodes[0], 1));
    try testing.expectEqual(@as(usize, 0), pathSum(&nodes[0], -1));
}
