//! Binary Tree Traversals - Zig implementation
//! Reference: https://github.com/TheAlgorithms/Python/blob/master/data_structures/binary_tree/binary_tree_traversals.py

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

pub fn makeTree(allocator: std.mem.Allocator) !*Node {
    const tree = try createNode(allocator, 1);
    errdefer freeTree(allocator, tree);

    tree.left = try createNode(allocator, 2);
    tree.right = try createNode(allocator, 3);
    tree.left.?.left = try createNode(allocator, 4);
    tree.left.?.right = try createNode(allocator, 5);

    return tree;
}

fn preorderRec(node: ?*const Node, out: *std.ArrayListUnmanaged(i64), allocator: std.mem.Allocator) !void {
    const n = node orelse return;
    try out.append(allocator, n.data);
    try preorderRec(n.left, out, allocator);
    try preorderRec(n.right, out, allocator);
}

pub fn preorder(allocator: std.mem.Allocator, root: ?*const Node) ![]i64 {
    var out = std.ArrayListUnmanaged(i64){};
    errdefer out.deinit(allocator);
    try preorderRec(root, &out, allocator);
    return out.toOwnedSlice(allocator);
}

fn postorderRec(node: ?*const Node, out: *std.ArrayListUnmanaged(i64), allocator: std.mem.Allocator) !void {
    const n = node orelse return;
    try postorderRec(n.left, out, allocator);
    try postorderRec(n.right, out, allocator);
    try out.append(allocator, n.data);
}

pub fn postorder(allocator: std.mem.Allocator, root: ?*const Node) ![]i64 {
    var out = std.ArrayListUnmanaged(i64){};
    errdefer out.deinit(allocator);
    try postorderRec(root, &out, allocator);
    return out.toOwnedSlice(allocator);
}

fn inorderRec(node: ?*const Node, out: *std.ArrayListUnmanaged(i64), allocator: std.mem.Allocator) !void {
    const n = node orelse return;
    try inorderRec(n.left, out, allocator);
    try out.append(allocator, n.data);
    try inorderRec(n.right, out, allocator);
}

pub fn inorder(allocator: std.mem.Allocator, root: ?*const Node) ![]i64 {
    var out = std.ArrayListUnmanaged(i64){};
    errdefer out.deinit(allocator);
    try inorderRec(root, &out, allocator);
    return out.toOwnedSlice(allocator);
}

fn reverseInorderRec(node: ?*const Node, out: *std.ArrayListUnmanaged(i64), allocator: std.mem.Allocator) !void {
    const n = node orelse return;
    try reverseInorderRec(n.right, out, allocator);
    try out.append(allocator, n.data);
    try reverseInorderRec(n.left, out, allocator);
}

pub fn reverseInorder(allocator: std.mem.Allocator, root: ?*const Node) ![]i64 {
    var out = std.ArrayListUnmanaged(i64){};
    errdefer out.deinit(allocator);
    try reverseInorderRec(root, &out, allocator);
    return out.toOwnedSlice(allocator);
}

pub fn height(root: ?*const Node) usize {
    const n = root orelse return 0;
    return @max(height(n.left), height(n.right)) + 1;
}

pub fn levelOrder(allocator: std.mem.Allocator, root: ?*const Node) ![]i64 {
    var out = std.ArrayListUnmanaged(i64){};
    errdefer out.deinit(allocator);

    const start = root orelse return out.toOwnedSlice(allocator);

    var queue = std.ArrayListUnmanaged(*const Node){};
    defer queue.deinit(allocator);
    try queue.append(allocator, start);

    var head: usize = 0;
    while (head < queue.items.len) {
        const node = queue.items[head];
        head += 1;

        try out.append(allocator, node.data);
        if (node.left) |left| try queue.append(allocator, left);
        if (node.right) |right| try queue.append(allocator, right);
    }

    return out.toOwnedSlice(allocator);
}

fn appendNodesLeftToRight(node: ?*const Node, level: usize, out: *std.ArrayListUnmanaged(i64), allocator: std.mem.Allocator) !void {
    const n = node orelse return;
    if (level == 1) {
        try out.append(allocator, n.data);
        return;
    }

    try appendNodesLeftToRight(n.left, level - 1, out, allocator);
    try appendNodesLeftToRight(n.right, level - 1, out, allocator);
}

pub fn getNodesFromLeftToRight(allocator: std.mem.Allocator, root: ?*const Node, level: usize) ![]i64 {
    var out = std.ArrayListUnmanaged(i64){};
    errdefer out.deinit(allocator);

    if (level == 0) return out.toOwnedSlice(allocator);

    try appendNodesLeftToRight(root, level, &out, allocator);
    return out.toOwnedSlice(allocator);
}

fn appendNodesRightToLeft(node: ?*const Node, level: usize, out: *std.ArrayListUnmanaged(i64), allocator: std.mem.Allocator) !void {
    const n = node orelse return;
    if (level == 1) {
        try out.append(allocator, n.data);
        return;
    }

    try appendNodesRightToLeft(n.right, level - 1, out, allocator);
    try appendNodesRightToLeft(n.left, level - 1, out, allocator);
}

pub fn getNodesFromRightToLeft(allocator: std.mem.Allocator, root: ?*const Node, level: usize) ![]i64 {
    var out = std.ArrayListUnmanaged(i64){};
    errdefer out.deinit(allocator);

    if (level == 0) return out.toOwnedSlice(allocator);

    try appendNodesRightToLeft(root, level, &out, allocator);
    return out.toOwnedSlice(allocator);
}

pub fn zigzag(allocator: std.mem.Allocator, root: ?*const Node) ![]i64 {
    var out = std.ArrayListUnmanaged(i64){};
    errdefer out.deinit(allocator);

    const h = height(root);
    var level: usize = 1;
    var left_to_right = true;

    while (level <= h) : (level += 1) {
        if (left_to_right) {
            try appendNodesLeftToRight(root, level, &out, allocator);
            left_to_right = false;
        } else {
            try appendNodesRightToLeft(root, level, &out, allocator);
            left_to_right = true;
        }
    }

    return out.toOwnedSlice(allocator);
}

test "binary tree traversals: python core traversals" {
    const root = try makeTree(testing.allocator);
    defer freeTree(testing.allocator, root);

    const pre = try preorder(testing.allocator, root);
    defer testing.allocator.free(pre);
    try testing.expectEqualSlices(i64, &[_]i64{ 1, 2, 4, 5, 3 }, pre);

    const post = try postorder(testing.allocator, root);
    defer testing.allocator.free(post);
    try testing.expectEqualSlices(i64, &[_]i64{ 4, 5, 2, 3, 1 }, post);

    const ino = try inorder(testing.allocator, root);
    defer testing.allocator.free(ino);
    try testing.expectEqualSlices(i64, &[_]i64{ 4, 2, 5, 1, 3 }, ino);

    const rev = try reverseInorder(testing.allocator, root);
    defer testing.allocator.free(rev);
    try testing.expectEqualSlices(i64, &[_]i64{ 3, 1, 5, 2, 4 }, rev);

    try testing.expectEqual(@as(usize, 3), height(root));
}

test "binary tree traversals: level/zigzag and level helpers" {
    const root = try makeTree(testing.allocator);
    defer freeTree(testing.allocator, root);

    const level = try levelOrder(testing.allocator, root);
    defer testing.allocator.free(level);
    try testing.expectEqualSlices(i64, &[_]i64{ 1, 2, 3, 4, 5 }, level);

    const l1 = try getNodesFromLeftToRight(testing.allocator, root, 1);
    defer testing.allocator.free(l1);
    try testing.expectEqualSlices(i64, &[_]i64{1}, l1);

    const l2 = try getNodesFromLeftToRight(testing.allocator, root, 2);
    defer testing.allocator.free(l2);
    try testing.expectEqualSlices(i64, &[_]i64{ 2, 3 }, l2);

    const r2 = try getNodesFromRightToLeft(testing.allocator, root, 2);
    defer testing.allocator.free(r2);
    try testing.expectEqualSlices(i64, &[_]i64{ 3, 2 }, r2);

    const zz = try zigzag(testing.allocator, root);
    defer testing.allocator.free(zz);
    try testing.expectEqualSlices(i64, &[_]i64{ 1, 3, 2, 4, 5 }, zz);
}

test "binary tree traversals: boundary and extreme" {
    const empty_pre = try preorder(testing.allocator, null);
    defer testing.allocator.free(empty_pre);
    try testing.expectEqual(@as(usize, 0), empty_pre.len);

    try testing.expectEqual(@as(usize, 0), height(null));

    const n: usize = 8_191;
    const nodes = try testing.allocator.alloc(Node, n);
    defer testing.allocator.free(nodes);

    for (0..n) |i| {
        nodes[i] = .{ .data = @intCast(i + 1) };
    }
    for (0..n) |i| {
        const left = i * 2 + 1;
        const right = i * 2 + 2;
        nodes[i].left = if (left < n) &nodes[left] else null;
        nodes[i].right = if (right < n) &nodes[right] else null;
    }

    try testing.expectEqual(@as(usize, 13), height(&nodes[0]));

    const level = try levelOrder(testing.allocator, &nodes[0]);
    defer testing.allocator.free(level);
    try testing.expectEqual(n, level.len);
    try testing.expectEqual(@as(i64, 1), level[0]);

    const zz = try zigzag(testing.allocator, &nodes[0]);
    defer testing.allocator.free(zz);
    try testing.expectEqual(n, zz.len);
}
