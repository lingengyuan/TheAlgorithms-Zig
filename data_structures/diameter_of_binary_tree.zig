//! Diameter Of Binary Tree - Zig implementation
//! Reference: https://github.com/TheAlgorithms/Python/blob/master/data_structures/binary_tree/diameter_of_binary_tree.py

const std = @import("std");
const testing = std.testing;

pub const Node = struct {
    data: i64,
    left: ?*Node = null,
    right: ?*Node = null,
};

/// Returns depth of subtree rooted at `node`.
/// Time complexity: O(n), Space complexity: O(h)
pub fn depth(node: ?*const Node) usize {
    const n = node orelse return 0;
    const left_depth = depth(n.left);
    const right_depth = depth(n.right);
    return @max(left_depth, right_depth) + 1;
}

/// Returns diameter of the subtree rooted at `node`.
/// Time complexity: O(n), Space complexity: O(h)
pub fn diameter(node: ?*const Node) usize {
    const n = node orelse return 0;
    const left_depth = depth(n.left);
    const right_depth = depth(n.right);
    return @max(left_depth + right_depth + 1, @max(diameter(n.left), diameter(n.right)));
}

test "diameter of binary tree: python doctest sequence" {
    var root = Node{ .data = 1 };
    try testing.expectEqual(@as(usize, 1), depth(&root));
    try testing.expectEqual(@as(usize, 1), diameter(&root));

    var n2 = Node{ .data = 2 };
    root.left = &n2;
    try testing.expectEqual(@as(usize, 2), depth(&root));
    try testing.expectEqual(@as(usize, 2), diameter(&root));
    try testing.expectEqual(@as(usize, 1), diameter(&n2));

    var n3 = Node{ .data = 3 };
    root.right = &n3;
    try testing.expectEqual(@as(usize, 2), depth(&root));
    try testing.expectEqual(@as(usize, 3), diameter(&root));
}

test "diameter of binary tree: comment sample" {
    var n1 = Node{ .data = 1 };
    var n2 = Node{ .data = 2 };
    var n3 = Node{ .data = 3 };
    var n4 = Node{ .data = 4 };
    var n5 = Node{ .data = 5 };

    n1.left = &n2;
    n1.right = &n3;
    n2.left = &n4;
    n2.right = &n5;

    try testing.expectEqual(@as(usize, 4), diameter(&n1));
    try testing.expectEqual(@as(usize, 3), diameter(&n2));
    try testing.expectEqual(@as(usize, 1), diameter(&n3));
}

test "diameter of binary tree: boundary and extreme" {
    try testing.expectEqual(@as(usize, 0), depth(null));
    try testing.expectEqual(@as(usize, 0), diameter(null));

    const n: usize = 3_000;
    const nodes = try testing.allocator.alloc(Node, n);
    defer testing.allocator.free(nodes);

    for (0..n) |i| {
        nodes[i] = .{ .data = @intCast(i) };
    }
    for (0..n) |i| {
        nodes[i].right = if (i + 1 < n) &nodes[i + 1] else null;
    }

    try testing.expectEqual(n, depth(&nodes[0]));
    try testing.expectEqual(n, diameter(&nodes[0]));
}

test "diameter of binary tree: subtree diameter can exceed root path" {
    var n1 = Node{ .data = 1 };
    var n2 = Node{ .data = 2 };
    var n3 = Node{ .data = 3 };
    var n4 = Node{ .data = 4 };
    var n5 = Node{ .data = 5 };
    var n6 = Node{ .data = 6 };

    n1.left = &n2;
    n2.left = &n3;
    n3.left = &n5;
    n2.right = &n4;
    n4.right = &n6;

    try testing.expectEqual(@as(usize, 5), diameter(&n1));
}
