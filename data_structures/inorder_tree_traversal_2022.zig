//! Inorder Tree Traversal 2022 - Zig implementation
//! Reference: https://github.com/TheAlgorithms/Python/blob/master/data_structures/binary_tree/inorder_tree_traversal_2022.py

const std = @import("std");
const testing = std.testing;

pub const BinaryTreeNode = struct {
    data: i64,
    left_child: ?*BinaryTreeNode = null,
    right_child: ?*BinaryTreeNode = null,
};

pub fn createNode(allocator: std.mem.Allocator, value: i64) !*BinaryTreeNode {
    const node = try allocator.create(BinaryTreeNode);
    node.* = .{ .data = value };
    return node;
}

pub fn freeTree(allocator: std.mem.Allocator, root: ?*BinaryTreeNode) void {
    const start = root orelse return;

    var stack = std.ArrayListUnmanaged(*BinaryTreeNode){};
    defer stack.deinit(allocator);
    stack.append(allocator, start) catch return;

    while (stack.items.len > 0) {
        const node = stack.pop().?;
        if (node.left_child) |left| stack.append(allocator, left) catch {};
        if (node.right_child) |right| stack.append(allocator, right) catch {};
        allocator.destroy(node);
    }
}

/// Inserts a value into BST rooted at `node` and returns (possibly same) root.
/// Time complexity: O(h), Space complexity: O(h)
pub fn insert(node: ?*BinaryTreeNode, new_value: i64, allocator: std.mem.Allocator) !*BinaryTreeNode {
    if (node == null) {
        return createNode(allocator, new_value);
    }

    if (new_value < node.?.data) {
        node.?.left_child = try insert(node.?.left_child, new_value, allocator);
    } else {
        node.?.right_child = try insert(node.?.right_child, new_value, allocator);
    }

    return node.?;
}

/// Returns inorder traversal values.
/// Time complexity: O(n), Space complexity: O(h)
pub fn inorder(allocator: std.mem.Allocator, root: ?*const BinaryTreeNode) ![]i64 {
    var out = std.ArrayListUnmanaged(i64){};
    errdefer out.deinit(allocator);

    var stack = std.ArrayListUnmanaged(*const BinaryTreeNode){};
    defer stack.deinit(allocator);

    var current = root;
    while (current != null or stack.items.len > 0) {
        while (current) |n| {
            try stack.append(allocator, n);
            current = n.left_child;
        }

        const node = stack.pop().?;
        try out.append(allocator, node.data);
        current = node.right_child;
    }

    return out.toOwnedSlice(allocator);
}

pub fn makeTree(allocator: std.mem.Allocator) !*BinaryTreeNode {
    var root = try insert(null, 15, allocator);
    root = try insert(root, 10, allocator);
    root = try insert(root, 25, allocator);
    root = try insert(root, 6, allocator);
    root = try insert(root, 14, allocator);
    root = try insert(root, 20, allocator);
    root = try insert(root, 60, allocator);
    return root;
}

test "inorder traversal 2022: python make_tree" {
    const root = try makeTree(testing.allocator);
    defer freeTree(testing.allocator, root);

    const values = try inorder(testing.allocator, root);
    defer testing.allocator.free(values);

    try testing.expectEqualSlices(i64, &[_]i64{ 6, 10, 14, 15, 20, 25, 60 }, values);
}

test "inorder traversal 2022: insert doctest behavior" {
    const node_a = try createNode(testing.allocator, 12_345);
    defer freeTree(testing.allocator, node_a);

    const node_b = try insert(node_a, 67_890, testing.allocator);

    try testing.expect(node_a == node_b);
    try testing.expectEqual(node_a.left_child, node_b.left_child);
    try testing.expectEqual(node_a.right_child, node_b.right_child);
    try testing.expectEqual(node_a.data, node_b.data);
}

test "inorder traversal 2022: boundary and extreme" {
    const empty = try inorder(testing.allocator, null);
    defer testing.allocator.free(empty);
    try testing.expectEqual(@as(usize, 0), empty.len);

    const n: usize = 40_000;
    const nodes = try testing.allocator.alloc(BinaryTreeNode, n);
    defer testing.allocator.free(nodes);

    for (0..n) |i| {
        nodes[i] = .{ .data = @intCast(i + 1) };
    }
    for (0..n) |i| {
        nodes[i].right_child = if (i + 1 < n) &nodes[i + 1] else null;
    }

    const values = try inorder(testing.allocator, &nodes[0]);
    defer testing.allocator.free(values);

    try testing.expectEqual(n, values.len);
    try testing.expectEqual(@as(i64, 1), values[0]);
    try testing.expectEqual(@as(i64, @intCast(n)), values[n - 1]);
}
