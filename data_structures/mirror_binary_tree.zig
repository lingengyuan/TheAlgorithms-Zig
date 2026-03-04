//! Mirror Binary Tree - Zig implementation
//! Reference: https://github.com/TheAlgorithms/Python/blob/master/data_structures/binary_tree/mirror_binary_tree.py

const std = @import("std");
const testing = std.testing;

pub const Node = struct {
    value: i64,
    left: ?*Node = null,
    right: ?*Node = null,
};

pub fn createNode(allocator: std.mem.Allocator, value: i64) !*Node {
    const node = try allocator.create(Node);
    node.* = .{ .value = value };
    return node;
}

pub fn freeTree(allocator: std.mem.Allocator, root: ?*Node) void {
    const start = root orelse return;

    var stack = std.ArrayListUnmanaged(*Node){};
    defer stack.deinit(allocator);
    stack.append(allocator, start) catch return;

    while (stack.items.len > 0) {
        const node = stack.pop().?;
        if (node.left) |left| stack.append(allocator, left) catch {};
        if (node.right) |right| stack.append(allocator, right) catch {};
        allocator.destroy(node);
    }
}

/// Mirrors tree in-place.
/// Time complexity: O(n), Space complexity: O(h)
pub fn mirror(root: *Node, allocator: std.mem.Allocator) !*Node {
    var stack = std.ArrayListUnmanaged(*Node){};
    defer stack.deinit(allocator);
    try stack.append(allocator, root);

    while (stack.items.len > 0) {
        const node = stack.pop().?;
        const tmp = node.left;
        node.left = node.right;
        node.right = tmp;

        if (node.left) |left| try stack.append(allocator, left);
        if (node.right) |right| try stack.append(allocator, right);
    }

    return root;
}

pub fn inorderValues(allocator: std.mem.Allocator, root: ?*const Node) ![]i64 {
    var out = std.ArrayListUnmanaged(i64){};
    errdefer out.deinit(allocator);

    var stack = std.ArrayListUnmanaged(*const Node){};
    defer stack.deinit(allocator);

    var current = root;
    while (current != null or stack.items.len > 0) {
        while (current) |n| {
            try stack.append(allocator, n);
            current = n.left;
        }

        const n = stack.pop().?;
        try out.append(allocator, n.value);
        current = n.right;
    }

    return out.toOwnedSlice(allocator);
}

pub fn makeTreeSeven(allocator: std.mem.Allocator) !*Node {
    const tree = try createNode(allocator, 1);
    errdefer freeTree(allocator, tree);

    tree.left = try createNode(allocator, 2);
    tree.right = try createNode(allocator, 3);
    tree.left.?.left = try createNode(allocator, 4);
    tree.left.?.right = try createNode(allocator, 5);
    tree.right.?.left = try createNode(allocator, 6);
    tree.right.?.right = try createNode(allocator, 7);

    return tree;
}

pub fn makeTreeNine(allocator: std.mem.Allocator) !*Node {
    const tree = try createNode(allocator, 1);
    errdefer freeTree(allocator, tree);

    tree.left = try createNode(allocator, 2);
    tree.right = try createNode(allocator, 3);
    tree.left.?.left = try createNode(allocator, 4);
    tree.left.?.right = try createNode(allocator, 5);
    tree.right.?.right = try createNode(allocator, 6);
    tree.left.?.left.?.left = try createNode(allocator, 7);
    tree.left.?.left.?.right = try createNode(allocator, 8);
    tree.left.?.right.?.right = try createNode(allocator, 9);

    return tree;
}

test "mirror binary tree: python node examples" {
    {
        const tree = try createNode(testing.allocator, 0);
        defer freeTree(testing.allocator, tree);

        const before = try inorderValues(testing.allocator, tree);
        defer testing.allocator.free(before);
        try testing.expectEqualSlices(i64, &[_]i64{0}, before);

        _ = try mirror(tree, testing.allocator);
        const after = try inorderValues(testing.allocator, tree);
        defer testing.allocator.free(after);
        try testing.expectEqualSlices(i64, &[_]i64{0}, after);
    }

    {
        const root = try createNode(testing.allocator, 1);
        defer freeTree(testing.allocator, root);
        root.left = try createNode(testing.allocator, 0);
        root.right = try createNode(testing.allocator, 3);
        root.right.?.left = try createNode(testing.allocator, 2);
        root.right.?.right = try createNode(testing.allocator, 4);
        root.right.?.right.?.right = try createNode(testing.allocator, 5);

        const before = try inorderValues(testing.allocator, root);
        defer testing.allocator.free(before);
        try testing.expectEqualSlices(i64, &[_]i64{ 0, 1, 2, 3, 4, 5 }, before);

        _ = try mirror(root, testing.allocator);
        const after = try inorderValues(testing.allocator, root);
        defer testing.allocator.free(after);
        try testing.expectEqualSlices(i64, &[_]i64{ 5, 4, 3, 2, 1, 0 }, after);
    }
}

test "mirror binary tree: python make_tree samples" {
    const seven = try makeTreeSeven(testing.allocator);
    defer freeTree(testing.allocator, seven);

    const seven_before = try inorderValues(testing.allocator, seven);
    defer testing.allocator.free(seven_before);
    try testing.expectEqualSlices(i64, &[_]i64{ 4, 2, 5, 1, 6, 3, 7 }, seven_before);

    _ = try mirror(seven, testing.allocator);
    const seven_after = try inorderValues(testing.allocator, seven);
    defer testing.allocator.free(seven_after);
    try testing.expectEqualSlices(i64, &[_]i64{ 7, 3, 6, 1, 5, 2, 4 }, seven_after);

    const nine = try makeTreeNine(testing.allocator);
    defer freeTree(testing.allocator, nine);

    const nine_before = try inorderValues(testing.allocator, nine);
    defer testing.allocator.free(nine_before);
    try testing.expectEqualSlices(i64, &[_]i64{ 7, 4, 8, 2, 5, 9, 1, 3, 6 }, nine_before);

    _ = try mirror(nine, testing.allocator);
    const nine_after = try inorderValues(testing.allocator, nine);
    defer testing.allocator.free(nine_after);
    try testing.expectEqualSlices(i64, &[_]i64{ 6, 3, 1, 9, 5, 2, 8, 4, 7 }, nine_after);
}

test "mirror binary tree: extreme mirror twice" {
    const n: usize = 40_000;
    const nodes = try testing.allocator.alloc(Node, n);
    defer testing.allocator.free(nodes);

    for (0..n) |i| {
        nodes[i] = .{ .value = @intCast(i) };
    }
    for (0..n) |i| {
        const left = i * 2 + 1;
        const right = i * 2 + 2;
        nodes[i].left = if (left < n) &nodes[left] else null;
        nodes[i].right = if (right < n) &nodes[right] else null;
    }

    const root = &nodes[0];
    const original = try inorderValues(testing.allocator, root);
    defer testing.allocator.free(original);

    _ = try mirror(root, testing.allocator);
    _ = try mirror(root, testing.allocator);

    const mirrored_twice = try inorderValues(testing.allocator, root);
    defer testing.allocator.free(mirrored_twice);

    try testing.expectEqualSlices(i64, original, mirrored_twice);
}
