//! Serialize/Deserialize Binary Tree - Zig implementation
//! Reference: https://github.com/TheAlgorithms/Python/blob/master/data_structures/binary_tree/serialize_deserialize_binary_tree.py

const std = @import("std");
const testing = std.testing;

pub const TreeNode = struct {
    value: i64,
    left: ?*TreeNode = null,
    right: ?*TreeNode = null,
};

pub fn createNode(allocator: std.mem.Allocator, value: i64) !*TreeNode {
    const node = try allocator.create(TreeNode);
    node.* = .{ .value = value };
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

fn appendSerialized(out: *std.ArrayListUnmanaged(u8), allocator: std.mem.Allocator, node: ?*const TreeNode) !void {
    if (node == null) {
        try out.appendSlice(allocator, "null");
        return;
    }

    var buf: [32]u8 = undefined;
    const n = try std.fmt.bufPrint(&buf, "{}", .{node.?.value});
    try out.appendSlice(allocator, n);
    try out.append(allocator, ',');
    try appendSerialized(out, allocator, node.?.left);
    try out.append(allocator, ',');
    try appendSerialized(out, allocator, node.?.right);
}

/// Serializes tree in preorder with null markers.
/// Time complexity: O(n), Space complexity: O(h)
pub fn serialize(allocator: std.mem.Allocator, root: ?*const TreeNode) ![]u8 {
    var out = std.ArrayListUnmanaged(u8){};
    errdefer out.deinit(allocator);

    try appendSerialized(&out, allocator, root);
    return out.toOwnedSlice(allocator);
}

fn buildFromTokens(allocator: std.mem.Allocator, tokens: []const []const u8, index: *usize) !?*TreeNode {
    if (index.* >= tokens.len) return error.InvalidSerializedData;

    const token = tokens[index.*];
    index.* += 1;

    if (std.mem.eql(u8, token, "null")) return null;

    const value = std.fmt.parseInt(i64, token, 10) catch return error.InvalidValue;

    const node = try allocator.create(TreeNode);
    errdefer allocator.destroy(node);

    node.* = .{ .value = value };
    node.left = try buildFromTokens(allocator, tokens, index);
    node.right = try buildFromTokens(allocator, tokens, index);
    return node;
}

/// Deserializes preorder/null-marker format into binary tree.
/// Time complexity: O(n), Space complexity: O(n)
pub fn deserialize(allocator: std.mem.Allocator, data: []const u8) !?*TreeNode {
    if (data.len == 0) return error.EmptyData;

    var tokens = std.ArrayListUnmanaged([]const u8){};
    defer tokens.deinit(allocator);

    var it = std.mem.splitScalar(u8, data, ',');
    while (it.next()) |token| {
        try tokens.append(allocator, token);
    }

    var index: usize = 0;
    const root = try buildFromTokens(allocator, tokens.items, &index);

    if (index != tokens.items.len) {
        freeTree(allocator, root);
        return error.InvalidSerializedData;
    }

    return root;
}

pub fn treeEqual(a: ?*const TreeNode, b: ?*const TreeNode) bool {
    if (a == null and b == null) return true;
    if (a == null or b == null) return false;

    return a.?.value == b.?.value and treeEqual(a.?.left, b.?.left) and treeEqual(a.?.right, b.?.right);
}

pub fn countNodes(allocator: std.mem.Allocator, root: ?*const TreeNode) !usize {
    const start = root orelse return 0;

    var stack = std.ArrayListUnmanaged(*const TreeNode){};
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

pub fn fiveTree(allocator: std.mem.Allocator) !*TreeNode {
    const root = try createNode(allocator, 1);
    errdefer freeTree(allocator, root);

    root.left = try createNode(allocator, 2);
    root.right = try createNode(allocator, 3);
    root.right.?.left = try createNode(allocator, 4);
    root.right.?.right = try createNode(allocator, 5);

    return root;
}

fn rightmostValue(root: ?*const TreeNode) i64 {
    var cur = root orelse return 0;
    while (cur.right) |r| cur = r;
    return cur.value;
}

test "serialize deserialize: python repr samples" {
    {
        var single = TreeNode{ .value = 1 };
        const repr = try serialize(testing.allocator, &single);
        defer testing.allocator.free(repr);
        try testing.expectEqualStrings("1,null,null", repr);
    }

    {
        var one = TreeNode{ .value = 1 };
        var two = TreeNode{ .value = 2 };
        var three = TreeNode{ .value = 3 };
        one.left = &two;
        one.right = &three;

        const repr = try serialize(testing.allocator, &one);
        defer testing.allocator.free(repr);
        try testing.expectEqualStrings("1,2,null,null,3,null,null", repr);
    }

    {
        const tree = try fiveTree(testing.allocator);
        defer freeTree(testing.allocator, tree);

        const repr = try serialize(testing.allocator, tree);
        defer testing.allocator.free(repr);
        try testing.expectEqualStrings("1,2,null,null,3,4,null,null,5,null,null", repr);
    }
}

test "serialize deserialize: roundtrip parity" {
    const root = try fiveTree(testing.allocator);
    defer freeTree(testing.allocator, root);

    const serialized = try serialize(testing.allocator, root);
    defer testing.allocator.free(serialized);

    const deserialized = (try deserialize(testing.allocator, serialized)).?;
    defer freeTree(testing.allocator, deserialized);

    try testing.expect(treeEqual(root, deserialized));
    try testing.expect(root != deserialized);

    root.right.?.right.?.value = 6;
    try testing.expect(!treeEqual(root, deserialized));

    const serialized2 = try serialize(testing.allocator, root);
    defer testing.allocator.free(serialized2);

    const deserialized2 = (try deserialize(testing.allocator, serialized2)).?;
    defer freeTree(testing.allocator, deserialized2);
    try testing.expect(treeEqual(root, deserialized2));
}

test "serialize deserialize: invalid input" {
    try testing.expectError(error.EmptyData, deserialize(testing.allocator, ""));
    try testing.expectError(error.InvalidValue, deserialize(testing.allocator, "1,x,null"));
    try testing.expectError(error.InvalidSerializedData, deserialize(testing.allocator, "1,null"));
}

test "serialize deserialize: extreme long chain" {
    const n: usize = 20_000;
    const nodes = try testing.allocator.alloc(TreeNode, n);
    defer testing.allocator.free(nodes);

    for (0..n) |i| {
        nodes[i] = .{ .value = @intCast(i + 1) };
    }
    for (0..n) |i| {
        nodes[i].right = if (i + 1 < n) &nodes[i + 1] else null;
    }

    const repr = try serialize(testing.allocator, &nodes[0]);
    defer testing.allocator.free(repr);

    const roundtrip = (try deserialize(testing.allocator, repr)).?;
    defer freeTree(testing.allocator, roundtrip);

    try testing.expectEqual(n, try countNodes(testing.allocator, roundtrip));
    try testing.expectEqual(@as(i64, 1), roundtrip.value);
    try testing.expectEqual(@as(i64, @intCast(n)), rightmostValue(roundtrip));
}
