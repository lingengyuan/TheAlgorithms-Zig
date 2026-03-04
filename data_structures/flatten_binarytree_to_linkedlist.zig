//! Flatten Binary Tree To Linked List - Zig implementation
//! Reference: https://github.com/TheAlgorithms/Python/blob/master/data_structures/binary_tree/flatten_binarytree_to_linkedlist.py

const std = @import("std");
const testing = std.testing;

pub const TreeNode = struct {
    data: i64,
    left: ?*TreeNode = null,
    right: ?*TreeNode = null,
};

pub fn createNode(allocator: std.mem.Allocator, data: i64) !*TreeNode {
    const node = try allocator.create(TreeNode);
    node.* = .{ .data = data };
    return node;
}

pub fn freeTree(allocator: std.mem.Allocator, root: ?*TreeNode) void {
    const start = root orelse return;

    var stack = std.ArrayListUnmanaged(*TreeNode){};
    defer stack.deinit(allocator);
    stack.append(allocator, start) catch return;

    while (stack.items.len > 0) {
        const node = stack.pop().?;
        if (node.left) |left| stack.append(allocator, left) catch {};
        if (node.right) |right| stack.append(allocator, right) catch {};
        allocator.destroy(node);
    }
}

pub fn buildTree(allocator: std.mem.Allocator) !*TreeNode {
    const root = try createNode(allocator, 1);
    errdefer freeTree(allocator, root);

    root.left = try createNode(allocator, 2);
    root.right = try createNode(allocator, 5);
    root.left.?.left = try createNode(allocator, 3);
    root.left.?.right = try createNode(allocator, 4);
    root.right.?.right = try createNode(allocator, 6);

    return root;
}

/// Flattens a binary tree into right-linked list in preorder order.
/// Time complexity: O(n), Space complexity: O(h)
pub fn flatten(root: ?*TreeNode, allocator: std.mem.Allocator) !void {
    const start = root orelse return;

    var stack = std.ArrayListUnmanaged(*TreeNode){};
    defer stack.deinit(allocator);
    try stack.append(allocator, start);

    var prev: ?*TreeNode = null;
    while (stack.items.len > 0) {
        const node = stack.pop().?;

        if (node.right) |right| try stack.append(allocator, right);
        if (node.left) |left| try stack.append(allocator, left);

        if (prev) |p| {
            p.left = null;
            p.right = node;
        }
        prev = node;
    }

    if (prev) |last| {
        last.left = null;
        last.right = null;
    }
}

pub fn linkedListValues(allocator: std.mem.Allocator, root: ?*const TreeNode) ![]i64 {
    var out = std.ArrayListUnmanaged(i64){};
    errdefer out.deinit(allocator);

    var current = root;
    while (current) |node| {
        try out.append(allocator, node.data);
        current = node.right;
    }

    return out.toOwnedSlice(allocator);
}

test "flatten binary tree: build tree sample" {
    const root = try buildTree(testing.allocator);
    defer freeTree(testing.allocator, root);

    try testing.expectEqual(@as(i64, 1), root.data);
    try testing.expectEqual(@as(i64, 2), root.left.?.data);
    try testing.expectEqual(@as(i64, 5), root.right.?.data);
    try testing.expectEqual(@as(i64, 3), root.left.?.left.?.data);
    try testing.expectEqual(@as(i64, 4), root.left.?.right.?.data);
    try testing.expectEqual(@as(i64, 6), root.right.?.right.?.data);
}

test "flatten binary tree: python doctest behavior" {
    const root = try buildTree(testing.allocator);
    defer freeTree(testing.allocator, root);

    try flatten(root, testing.allocator);

    const values = try linkedListValues(testing.allocator, root);
    defer testing.allocator.free(values);
    try testing.expectEqualSlices(i64, &[_]i64{ 1, 2, 3, 4, 5, 6 }, values);

    var cur: ?*TreeNode = root;
    while (cur) |node| {
        try testing.expect(node.left == null);
        cur = node.right;
    }
}

test "flatten binary tree: display linked list samples" {
    var n1 = TreeNode{ .data = 1 };
    var n2 = TreeNode{ .data = 2 };
    var n3 = TreeNode{ .data = 3 };
    n1.right = &n2;
    n2.right = &n3;

    const values = try linkedListValues(testing.allocator, &n1);
    defer testing.allocator.free(values);
    try testing.expectEqualSlices(i64, &[_]i64{ 1, 2, 3 }, values);

    const empty = try linkedListValues(testing.allocator, null);
    defer testing.allocator.free(empty);
    try testing.expectEqual(@as(usize, 0), empty.len);
}

test "flatten binary tree: extreme deep left chain" {
    const n: usize = 30_000;
    const nodes = try testing.allocator.alloc(TreeNode, n);
    defer testing.allocator.free(nodes);

    for (0..n) |i| {
        nodes[i] = .{ .data = @intCast(i) };
    }
    for (0..n) |i| {
        nodes[i].left = if (i + 1 < n) &nodes[i + 1] else null;
    }

    try flatten(&nodes[0], testing.allocator);

    var cur: ?*TreeNode = &nodes[0];
    var idx: usize = 0;
    while (cur) |node| : (idx += 1) {
        try testing.expect(node.left == null);
        try testing.expectEqual(@as(i64, @intCast(idx)), node.data);
        cur = node.right;
    }
    try testing.expectEqual(n, idx);
}
