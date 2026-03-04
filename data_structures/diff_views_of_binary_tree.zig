//! Different Views Of Binary Tree - Zig implementation
//! Reference: https://github.com/TheAlgorithms/Python/blob/master/data_structures/binary_tree/diff_views_of_binary_tree.py

const std = @import("std");
const testing = std.testing;

pub const TreeNode = struct {
    val: i64,
    left: ?*TreeNode = null,
    right: ?*TreeNode = null,
};

pub fn makeTree(allocator: std.mem.Allocator) !*TreeNode {
    const root = try allocator.create(TreeNode);
    errdefer allocator.destroy(root);
    root.* = .{ .val = 3 };

    root.left = try allocator.create(TreeNode);
    root.left.?.* = .{ .val = 9 };

    root.right = try allocator.create(TreeNode);
    root.right.?.* = .{ .val = 20 };

    root.right.?.left = try allocator.create(TreeNode);
    root.right.?.left.?.* = .{ .val = 15 };

    root.right.?.right = try allocator.create(TreeNode);
    root.right.?.right.?.* = .{ .val = 7 };

    return root;
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

fn rightDfs(node: ?*const TreeNode, depth: usize, out: *std.ArrayListUnmanaged(i64), allocator: std.mem.Allocator) !void {
    const n = node orelse return;
    if (depth == out.items.len) try out.append(allocator, n.val);
    try rightDfs(n.right, depth + 1, out, allocator);
    try rightDfs(n.left, depth + 1, out, allocator);
}

pub fn binaryTreeRightSideView(allocator: std.mem.Allocator, root: ?*const TreeNode) ![]i64 {
    var out = std.ArrayListUnmanaged(i64){};
    errdefer out.deinit(allocator);
    try rightDfs(root, 0, &out, allocator);
    return out.toOwnedSlice(allocator);
}

fn leftDfs(node: ?*const TreeNode, depth: usize, out: *std.ArrayListUnmanaged(i64), allocator: std.mem.Allocator) !void {
    const n = node orelse return;
    if (depth == out.items.len) try out.append(allocator, n.val);
    try leftDfs(n.left, depth + 1, out, allocator);
    try leftDfs(n.right, depth + 1, out, allocator);
}

pub fn binaryTreeLeftSideView(allocator: std.mem.Allocator, root: ?*const TreeNode) ![]i64 {
    var out = std.ArrayListUnmanaged(i64){};
    errdefer out.deinit(allocator);
    try leftDfs(root, 0, &out, allocator);
    return out.toOwnedSlice(allocator);
}

fn lessI64(_: void, a: i64, b: i64) bool {
    return a < b;
}

pub fn binaryTreeTopSideView(allocator: std.mem.Allocator, root: ?*const TreeNode) ![]i64 {
    var out = std.ArrayListUnmanaged(i64){};
    errdefer out.deinit(allocator);

    const start = root orelse return out.toOwnedSlice(allocator);

    var queue = std.ArrayListUnmanaged(struct { node: *const TreeNode, hd: i64 }){};
    defer queue.deinit(allocator);
    try queue.append(allocator, .{ .node = start, .hd = 0 });

    var lookup = std.AutoHashMap(i64, i64).init(allocator);
    defer lookup.deinit();

    var head: usize = 0;
    while (head < queue.items.len) {
        const item = queue.items[head];
        head += 1;

        if (!lookup.contains(item.hd)) {
            try lookup.put(item.hd, item.node.val);
        }

        if (item.node.left) |left| try queue.append(allocator, .{ .node = left, .hd = item.hd - 1 });
        if (item.node.right) |right| try queue.append(allocator, .{ .node = right, .hd = item.hd + 1 });
    }

    const keys = try allocator.alloc(i64, lookup.count());
    defer allocator.free(keys);

    var idx: usize = 0;
    var it = lookup.iterator();
    while (it.next()) |entry| : (idx += 1) {
        keys[idx] = entry.key_ptr.*;
    }
    std.mem.sort(i64, keys, {}, lessI64);

    for (keys) |k| {
        try out.append(allocator, lookup.get(k).?);
    }

    return out.toOwnedSlice(allocator);
}

pub fn binaryTreeBottomSideView(allocator: std.mem.Allocator, root: ?*const TreeNode) ![]i64 {
    var out = std.ArrayListUnmanaged(i64){};
    errdefer out.deinit(allocator);

    const start = root orelse return out.toOwnedSlice(allocator);

    var queue = std.ArrayListUnmanaged(struct { node: *const TreeNode, hd: i64 }){};
    defer queue.deinit(allocator);
    try queue.append(allocator, .{ .node = start, .hd = 0 });

    var lookup = std.AutoHashMap(i64, i64).init(allocator);
    defer lookup.deinit();

    var head: usize = 0;
    while (head < queue.items.len) {
        const item = queue.items[head];
        head += 1;

        try lookup.put(item.hd, item.node.val);

        if (item.node.left) |left| try queue.append(allocator, .{ .node = left, .hd = item.hd - 1 });
        if (item.node.right) |right| try queue.append(allocator, .{ .node = right, .hd = item.hd + 1 });
    }

    const keys = try allocator.alloc(i64, lookup.count());
    defer allocator.free(keys);

    var idx: usize = 0;
    var it = lookup.iterator();
    while (it.next()) |entry| : (idx += 1) {
        keys[idx] = entry.key_ptr.*;
    }
    std.mem.sort(i64, keys, {}, lessI64);

    for (keys) |k| {
        try out.append(allocator, lookup.get(k).?);
    }

    return out.toOwnedSlice(allocator);
}

test "different views of binary tree: python examples" {
    const root = try makeTree(testing.allocator);
    defer freeTree(testing.allocator, root);

    const right = try binaryTreeRightSideView(testing.allocator, root);
    defer testing.allocator.free(right);
    try testing.expectEqualSlices(i64, &[_]i64{ 3, 20, 7 }, right);

    const left = try binaryTreeLeftSideView(testing.allocator, root);
    defer testing.allocator.free(left);
    try testing.expectEqualSlices(i64, &[_]i64{ 3, 9, 15 }, left);

    const top = try binaryTreeTopSideView(testing.allocator, root);
    defer testing.allocator.free(top);
    try testing.expectEqualSlices(i64, &[_]i64{ 9, 3, 20, 7 }, top);

    const bottom = try binaryTreeBottomSideView(testing.allocator, root);
    defer testing.allocator.free(bottom);
    try testing.expectEqualSlices(i64, &[_]i64{ 9, 15, 20, 7 }, bottom);
}

test "different views of binary tree: empty" {
    const right = try binaryTreeRightSideView(testing.allocator, null);
    defer testing.allocator.free(right);
    try testing.expectEqual(@as(usize, 0), right.len);

    const left = try binaryTreeLeftSideView(testing.allocator, null);
    defer testing.allocator.free(left);
    try testing.expectEqual(@as(usize, 0), left.len);

    const top = try binaryTreeTopSideView(testing.allocator, null);
    defer testing.allocator.free(top);
    try testing.expectEqual(@as(usize, 0), top.len);

    const bottom = try binaryTreeBottomSideView(testing.allocator, null);
    defer testing.allocator.free(bottom);
    try testing.expectEqual(@as(usize, 0), bottom.len);
}

test "different views of binary tree: extreme complete tree" {
    const depth: usize = 11;
    const n: usize = (@as(usize, 1) << depth) - 1;

    const nodes = try testing.allocator.alloc(TreeNode, n);
    defer testing.allocator.free(nodes);

    for (0..n) |i| {
        nodes[i] = .{ .val = @intCast(i + 1) };
    }
    for (0..n) |i| {
        const left = i * 2 + 1;
        const right = i * 2 + 2;
        nodes[i].left = if (left < n) &nodes[left] else null;
        nodes[i].right = if (right < n) &nodes[right] else null;
    }

    const right = try binaryTreeRightSideView(testing.allocator, &nodes[0]);
    defer testing.allocator.free(right);
    try testing.expectEqual(depth, right.len);

    const left = try binaryTreeLeftSideView(testing.allocator, &nodes[0]);
    defer testing.allocator.free(left);
    try testing.expectEqual(depth, left.len);

    const top = try binaryTreeTopSideView(testing.allocator, &nodes[0]);
    defer testing.allocator.free(top);
    try testing.expectEqual(depth * 2 - 1, top.len);

    const bottom = try binaryTreeBottomSideView(testing.allocator, &nodes[0]);
    defer testing.allocator.free(bottom);
    try testing.expectEqual(depth * 2 - 1, bottom.len);
}
