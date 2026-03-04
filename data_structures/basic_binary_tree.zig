//! Basic Binary Tree - Zig implementation
//! Reference: https://github.com/TheAlgorithms/Python/blob/master/data_structures/binary_tree/basic_binary_tree.py

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

/// In-order traversal values.
/// Time complexity: O(n), Space complexity: O(h)
pub fn inorderTraversal(allocator: std.mem.Allocator, root: ?*const Node) ![]i64 {
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

/// Returns the number of nodes in the tree.
/// Time complexity: O(n), Space complexity: O(h)
pub fn treeLen(allocator: std.mem.Allocator, root: ?*const Node) !usize {
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

/// Returns tree depth.
/// Time complexity: O(n), Space complexity: O(w)
pub fn treeDepth(allocator: std.mem.Allocator, root: ?*const Node) !usize {
    const start = root orelse return 0;

    var queue = std.ArrayListUnmanaged(*const Node){};
    defer queue.deinit(allocator);
    try queue.append(allocator, start);

    var head: usize = 0;
    var depth: usize = 0;

    while (head < queue.items.len) {
        const level_end = queue.items.len;
        while (head < level_end) {
            const node = queue.items[head];
            head += 1;

            if (node.left) |left| try queue.append(allocator, left);
            if (node.right) |right| try queue.append(allocator, right);
        }
        depth += 1;
    }

    return depth;
}

/// Returns true if the tree is full.
/// Time complexity: O(n), Space complexity: O(h)
pub fn isFullTree(allocator: std.mem.Allocator, root: ?*const Node) !bool {
    const start = root orelse return true;

    var stack = std.ArrayListUnmanaged(*const Node){};
    defer stack.deinit(allocator);
    try stack.append(allocator, start);

    while (stack.items.len > 0) {
        const node = stack.pop().?;

        if ((node.left == null) != (node.right == null)) return false;

        if (node.left) |left| try stack.append(allocator, left);
        if (node.right) |right| try stack.append(allocator, right);
    }

    return true;
}

pub fn buildSmallTree(allocator: std.mem.Allocator) !*Node {
    const root = try createNode(allocator, 2);
    errdefer freeTree(allocator, root);

    root.left = try createNode(allocator, 1);
    root.right = try createNode(allocator, 3);
    return root;
}

pub fn buildMediumTree(allocator: std.mem.Allocator) !*Node {
    const root = try createNode(allocator, 4);
    errdefer freeTree(allocator, root);

    const two = try createNode(allocator, 2);
    const five = try createNode(allocator, 5);
    root.left = two;
    root.right = five;

    two.left = try createNode(allocator, 1);
    two.right = try createNode(allocator, 3);

    const six = try createNode(allocator, 6);
    five.right = six;
    six.right = try createNode(allocator, 7);

    return root;
}

test "basic binary tree: small tree python sample" {
    const root = try buildSmallTree(testing.allocator);
    defer freeTree(testing.allocator, root);

    const values = try inorderTraversal(testing.allocator, root);
    defer testing.allocator.free(values);

    try testing.expectEqualSlices(i64, &[_]i64{ 1, 2, 3 }, values);
    try testing.expectEqual(@as(usize, 3), try treeLen(testing.allocator, root));
    try testing.expectEqual(@as(usize, 2), try treeDepth(testing.allocator, root));
    try testing.expect(try isFullTree(testing.allocator, root));
}

test "basic binary tree: medium tree python sample" {
    const root = try buildMediumTree(testing.allocator);
    defer freeTree(testing.allocator, root);

    const values = try inorderTraversal(testing.allocator, root);
    defer testing.allocator.free(values);

    try testing.expectEqualSlices(i64, &[_]i64{ 1, 2, 3, 4, 5, 6, 7 }, values);
    try testing.expectEqual(@as(usize, 7), try treeLen(testing.allocator, root));
    try testing.expectEqual(@as(usize, 4), try treeDepth(testing.allocator, root));
    try testing.expect(!(try isFullTree(testing.allocator, root)));
}

test "basic binary tree: boundary" {
    const single = try createNode(testing.allocator, 1);
    defer freeTree(testing.allocator, single);

    const values = try inorderTraversal(testing.allocator, single);
    defer testing.allocator.free(values);

    try testing.expectEqualSlices(i64, &[_]i64{1}, values);
    try testing.expectEqual(@as(usize, 1), try treeLen(testing.allocator, single));
    try testing.expectEqual(@as(usize, 1), try treeDepth(testing.allocator, single));
    try testing.expect(try isFullTree(testing.allocator, single));

    try testing.expectEqual(@as(usize, 0), try treeLen(testing.allocator, null));
    try testing.expectEqual(@as(usize, 0), try treeDepth(testing.allocator, null));
    try testing.expect(try isFullTree(testing.allocator, null));
}

test "basic binary tree: extreme skewed tree" {
    const n: usize = 30_000;

    const root = try createNode(testing.allocator, 0);
    defer freeTree(testing.allocator, root);

    var cur = root;
    var i: usize = 1;
    while (i < n) : (i += 1) {
        const node = try createNode(testing.allocator, @intCast(i));
        cur.right = node;
        cur = node;
    }

    try testing.expectEqual(n, try treeLen(testing.allocator, root));
    try testing.expectEqual(n, try treeDepth(testing.allocator, root));
    try testing.expect(!(try isFullTree(testing.allocator, root)));

    const values = try inorderTraversal(testing.allocator, root);
    defer testing.allocator.free(values);

    try testing.expectEqual(n, values.len);
    try testing.expectEqual(@as(i64, 0), values[0]);
    try testing.expectEqual(@as(i64, @intCast(n - 1)), values[n - 1]);
}
