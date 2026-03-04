//! Symmetric Tree - Zig implementation
//! Reference: https://github.com/TheAlgorithms/Python/blob/master/data_structures/binary_tree/symmetric_tree.py

const std = @import("std");
const testing = std.testing;

pub const Node = struct {
    data: i64,
    left: ?*Node = null,
    right: ?*Node = null,
};

pub fn createNode(allocator: std.mem.Allocator, data: i64) !*Node {
    const node = try allocator.create(Node);
    node.* = .{ .data = data };
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

pub fn makeSymmetricTree(allocator: std.mem.Allocator) !*Node {
    const root = try createNode(allocator, 1);
    errdefer freeTree(allocator, root);

    root.left = try createNode(allocator, 2);
    root.right = try createNode(allocator, 2);
    root.left.?.left = try createNode(allocator, 3);
    root.left.?.right = try createNode(allocator, 4);
    root.right.?.left = try createNode(allocator, 4);
    root.right.?.right = try createNode(allocator, 3);

    return root;
}

pub fn makeAsymmetricTree(allocator: std.mem.Allocator) !*Node {
    const root = try createNode(allocator, 1);
    errdefer freeTree(allocator, root);

    root.left = try createNode(allocator, 2);
    root.right = try createNode(allocator, 2);
    root.left.?.left = try createNode(allocator, 3);
    root.left.?.right = try createNode(allocator, 4);
    root.right.?.left = try createNode(allocator, 3);
    root.right.?.right = try createNode(allocator, 4);

    return root;
}

/// Returns true if two trees are mirrors.
/// Time complexity: O(n), Space complexity: O(h)
pub fn isMirror(left: ?*const Node, right: ?*const Node) bool {
    if (left == null and right == null) return true;
    if (left == null or right == null) return false;

    return left.?.data == right.?.data and isMirror(left.?.left, right.?.right) and isMirror(left.?.right, right.?.left);
}

/// Returns true if tree is symmetric around center.
/// Time complexity: O(n), Space complexity: O(h)
pub fn isSymmetricTree(tree: ?*const Node) bool {
    const root = tree orelse return true;
    return isMirror(root.left, root.right);
}

test "symmetric tree: python examples" {
    const symmetric = try makeSymmetricTree(testing.allocator);
    defer freeTree(testing.allocator, symmetric);
    try testing.expect(isSymmetricTree(symmetric));
    try testing.expect(isMirror(symmetric.left, symmetric.right));

    const asymmetric = try makeAsymmetricTree(testing.allocator);
    defer freeTree(testing.allocator, asymmetric);
    try testing.expect(!isSymmetricTree(asymmetric));
    try testing.expect(!isMirror(asymmetric.left, asymmetric.right));
}

test "symmetric tree: boundary" {
    try testing.expect(isSymmetricTree(null));

    var single = Node{ .data = 42 };
    try testing.expect(isSymmetricTree(&single));

    var left_only = Node{ .data = 1 };
    var left_child = Node{ .data = 2 };
    left_only.left = &left_child;
    try testing.expect(!isSymmetricTree(&left_only));
}

test "symmetric tree: extreme perfect tree" {
    const depth: usize = 13;
    const n: usize = (@as(usize, 1) << depth) - 1;

    const nodes = try testing.allocator.alloc(Node, n);
    defer testing.allocator.free(nodes);

    for (0..n) |i| {
        nodes[i] = .{ .data = 1 };
    }
    for (0..n) |i| {
        const left_idx = i * 2 + 1;
        const right_idx = i * 2 + 2;
        nodes[i].left = if (left_idx < n) &nodes[left_idx] else null;
        nodes[i].right = if (right_idx < n) &nodes[right_idx] else null;
    }

    try testing.expect(isSymmetricTree(&nodes[0]));

    nodes[n - 1].data = 2;
    try testing.expect(!isSymmetricTree(&nodes[0]));
}
