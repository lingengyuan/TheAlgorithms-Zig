//! Treap - Zig implementation
//! Reference: https://github.com/TheAlgorithms/Python/blob/master/data_structures/binary_tree/treap.py

const std = @import("std");
const testing = std.testing;

pub const Node = struct {
    value: i64,
    prior: f64,
    left: ?*Node,
    right: ?*Node,
};

fn split(root: ?*Node, value: i64) struct { left: ?*Node, right: ?*Node } {
    if (root == null) return .{ .left = null, .right = null };

    var r = root.?;
    if (value < r.value) {
        const res = split(r.left, value);
        r.left = res.right;
        return .{ .left = res.left, .right = r };
    } else {
        const res = split(r.right, value);
        r.right = res.left;
        return .{ .left = r, .right = res.right };
    }
}

fn merge(left: ?*Node, right: ?*Node) ?*Node {
    if (left == null) return right;
    if (right == null) return left;

    var l = left.?;
    var r = right.?;
    if (l.prior < r.prior) {
        l.right = merge(l.right, r);
        return l;
    } else {
        r.left = merge(l, r.left);
        return r;
    }
}

pub const Treap = struct {
    allocator: std.mem.Allocator,
    root: ?*Node,
    prng: std.Random.DefaultPrng,

    pub fn init(allocator: std.mem.Allocator, seed: u64) Treap {
        return .{
            .allocator = allocator,
            .root = null,
            .prng = std.Random.DefaultPrng.init(seed),
        };
    }

    pub fn deinit(self: *Treap) void {
        self.freeSubtree(self.root);
        self.* = undefined;
    }

    fn freeSubtree(self: *Treap, node: ?*Node) void {
        if (node == null) return;
        const n = node.?;
        self.freeSubtree(n.left);
        self.freeSubtree(n.right);
        self.allocator.destroy(n);
    }

    fn makeNode(self: *Treap, value: i64) !*Node {
        const node = try self.allocator.create(Node);
        node.* = .{
            .value = value,
            .prior = self.prng.random().float(f64),
            .left = null,
            .right = null,
        };
        return node;
    }

    /// Inserts value (duplicates allowed).
    /// Time complexity: O(log n) expected, Space complexity: O(1) extra.
    pub fn insert(self: *Treap, value: i64) !void {
        const node = try self.makeNode(value);
        const res = split(self.root, value);
        self.root = merge(merge(res.left, node), res.right);
    }

    /// Removes all nodes equal to value.
    /// Time complexity: O(log n) expected, Space complexity: O(1) extra.
    pub fn erase(self: *Treap, value: i64) void {
        const a = split(self.root, value - 1);
        const b = split(a.right, value);
        self.freeSubtree(b.left);
        self.root = merge(a.left, b.right);
    }

    pub fn inorder(self: *const Treap, allocator: std.mem.Allocator) ![]i64 {
        var list = std.ArrayListUnmanaged(i64){};
        errdefer list.deinit(allocator);

        try inorderNode(self.root, &list, allocator);
        return try list.toOwnedSlice(allocator);
    }

    fn inorderNode(node: ?*Node, out: *std.ArrayListUnmanaged(i64), allocator: std.mem.Allocator) !void {
        if (node == null) return;
        const n = node.?;
        try inorderNode(n.left, out, allocator);
        try out.append(allocator, n.value);
        try inorderNode(n.right, out, allocator);
    }

    /// Processes command string with tokens like "+4 -2".
    pub fn interactTreap(self: *Treap, args: []const u8) !void {
        var it = std.mem.tokenizeScalar(u8, args, ' ');
        while (it.next()) |arg| {
            if (arg.len < 2) continue;
            if (arg[0] == '+') {
                const value = try std.fmt.parseInt(i64, arg[1..], 10);
                try self.insert(value);
            } else if (arg[0] == '-') {
                const value = try std.fmt.parseInt(i64, arg[1..], 10);
                self.erase(value);
            }
        }
    }
};

test "treap: python command sample" {
    var treap = Treap.init(testing.allocator, 1);
    defer treap.deinit();

    try treap.interactTreap("+1");
    var values = try treap.inorder(testing.allocator);
    defer testing.allocator.free(values);
    try testing.expectEqualSlices(i64, &[_]i64{1}, values);

    try treap.interactTreap("+3 +5 +17 +19 +2 +16 +4 +0");
    testing.allocator.free(values);
    values = try treap.inorder(testing.allocator);
    try testing.expectEqualSlices(i64, &[_]i64{ 0, 1, 2, 3, 4, 5, 16, 17, 19 }, values);

    try treap.interactTreap("+4 +4 +4");
    testing.allocator.free(values);
    values = try treap.inorder(testing.allocator);
    try testing.expectEqualSlices(i64, &[_]i64{ 0, 1, 2, 3, 4, 4, 4, 4, 5, 16, 17, 19 }, values);

    try treap.interactTreap("-0");
    testing.allocator.free(values);
    values = try treap.inorder(testing.allocator);
    try testing.expectEqualSlices(i64, &[_]i64{ 1, 2, 3, 4, 4, 4, 4, 5, 16, 17, 19 }, values);

    try treap.interactTreap("-4");
    testing.allocator.free(values);
    values = try treap.inorder(testing.allocator);
    try testing.expectEqualSlices(i64, &[_]i64{ 1, 2, 3, 5, 16, 17, 19 }, values);
}

test "treap: erase from empty and non-existing" {
    var treap = Treap.init(testing.allocator, 2);
    defer treap.deinit();

    treap.erase(10);
    try treap.insert(3);
    try treap.insert(1);
    treap.erase(99);

    const values = try treap.inorder(testing.allocator);
    defer testing.allocator.free(values);
    try testing.expectEqualSlices(i64, &[_]i64{ 1, 3 }, values);
}

test "treap: extreme insert/erase" {
    var treap = Treap.init(testing.allocator, 12345);
    defer treap.deinit();

    for (0..20_000) |i| {
        try treap.insert(@intCast(i % 1000));
    }

    for (0..1000) |i| {
        treap.erase(@intCast(i));
    }

    const values = try treap.inorder(testing.allocator);
    defer testing.allocator.free(values);
    try testing.expectEqual(@as(usize, 0), values.len);
}
