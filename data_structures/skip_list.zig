//! Skip List - Zig implementation
//! Reference: https://github.com/TheAlgorithms/Python/blob/master/data_structures/linked_list/skip_list.py

const std = @import("std");
const testing = std.testing;

pub const SkipListError = error{ InvalidProbability, InvalidMaxLevel };

const Node = struct {
    key: i64,
    value: i64,
    forward: []?*Node,
};

pub const SkipList = struct {
    allocator: std.mem.Allocator,
    head: *Node,
    level: usize,
    p: f64,
    max_level: usize,
    prng: std.Random.DefaultPrng,

    pub fn init(allocator: std.mem.Allocator, p: f64, max_level: usize, seed: u64) !SkipList {
        if (!(p > 0 and p < 1)) return SkipListError.InvalidProbability;
        if (max_level == 0) return SkipListError.InvalidMaxLevel;

        const head = try createNode(allocator, max_level, std.math.minInt(i64), 0);
        return .{
            .allocator = allocator,
            .head = head,
            .level = 1,
            .p = p,
            .max_level = max_level,
            .prng = std.Random.DefaultPrng.init(seed),
        };
    }

    pub fn deinit(self: *SkipList) void {
        var node_opt = self.head.forward[0];
        while (node_opt) |node| {
            const next = node.forward[0];
            self.allocator.free(node.forward);
            self.allocator.destroy(node);
            node_opt = next;
        }
        self.allocator.free(self.head.forward);
        self.allocator.destroy(self.head);
        self.* = undefined;
    }

    fn createNode(allocator: std.mem.Allocator, level: usize, key: i64, value: i64) !*Node {
        const node = try allocator.create(Node);
        errdefer allocator.destroy(node);

        const forward = try allocator.alloc(?*Node, level);
        errdefer allocator.free(forward);
        @memset(forward, null);

        node.* = .{ .key = key, .value = value, .forward = forward };
        return node;
    }

    fn randomLevel(self: *SkipList) usize {
        var lvl: usize = 1;
        const rnd = self.prng.random();
        while (rnd.float(f64) < self.p and lvl < self.max_level) {
            lvl += 1;
        }
        return lvl;
    }

    fn locateNode(self: *SkipList, key: i64, update: []?*Node) ?*Node {
        var node = self.head;
        var lvl = self.level;
        while (lvl > 0) {
            lvl -= 1;
            while (lvl < node.forward.len and node.forward[lvl] != null and node.forward[lvl].?.key < key) {
                node = node.forward[lvl].?;
            }
            update[lvl] = node;
        }

        const candidate = node.forward[0];
        if (candidate != null and candidate.?.key == key) return candidate.?;
        return null;
    }

    /// Inserts key/value; updates existing value for duplicate key.
    /// Time complexity: O(log n) expected, Space complexity: O(1) extra.
    pub fn insert(self: *SkipList, key: i64, value: i64) !void {
        const update = try self.allocator.alloc(?*Node, self.max_level);
        defer self.allocator.free(update);
        @memset(update, null);

        const found = self.locateNode(key, update);
        if (found) |node| {
            node.value = value;
            return;
        }

        const lvl = self.randomLevel();
        if (lvl > self.level) {
            for (self.level..lvl) |i| {
                update[i] = self.head;
            }
            self.level = lvl;
        }

        const new_node = try createNode(self.allocator, lvl, key, value);

        for (0..lvl) |i| {
            new_node.forward[i] = update[i].?.forward[i];
            update[i].?.forward[i] = new_node;
        }
    }

    /// Deletes key if present.
    /// Time complexity: O(log n) expected, Space complexity: O(1) extra.
    pub fn delete(self: *SkipList, key: i64) !void {
        const update = try self.allocator.alloc(?*Node, self.max_level);
        defer self.allocator.free(update);
        @memset(update, null);

        const found = self.locateNode(key, update);
        if (found == null) return;
        const node = found.?;

        for (0..self.level) |i| {
            if (i >= update[i].?.forward.len) continue;
            if (update[i].?.forward[i] == node) {
                update[i].?.forward[i] = if (i < node.forward.len) node.forward[i] else null;
            }
        }

        self.allocator.free(node.forward);
        self.allocator.destroy(node);

        while (self.level > 1 and self.head.forward[self.level - 1] == null) {
            self.level -= 1;
        }
    }

    pub fn find(self: *SkipList, key: i64) !?i64 {
        const update = try self.allocator.alloc(?*Node, self.max_level);
        defer self.allocator.free(update);
        @memset(update, null);

        const found = self.locateNode(key, update);
        return if (found) |node| node.value else null;
    }

    pub fn keys(self: *const SkipList, allocator: std.mem.Allocator) ![]i64 {
        var list = std.ArrayListUnmanaged(i64){};
        errdefer list.deinit(allocator);

        var node_opt = self.head.forward[0];
        while (node_opt) |node| {
            try list.append(allocator, node.key);
            node_opt = node.forward[0];
        }

        return try list.toOwnedSlice(allocator);
    }
};

test "skip list: insert and find" {
    var sl = try SkipList.init(testing.allocator, 0.5, 16, 1);
    defer sl.deinit();

    try sl.insert(1, 3);
    try sl.insert(2, 12);
    try sl.insert(3, 41);
    try sl.insert(4, -19);

    try testing.expectEqual(@as(?i64, 3), try sl.find(1));
    try testing.expectEqual(@as(?i64, 12), try sl.find(2));
    try testing.expectEqual(@as(?i64, 41), try sl.find(3));
    try testing.expectEqual(@as(?i64, -19), try sl.find(4));
}

test "skip list: override existing values" {
    var sl = try SkipList.init(testing.allocator, 0.5, 16, 2);
    defer sl.deinit();

    try sl.insert(1, 10);
    try sl.insert(1, 12);
    try sl.insert(5, 7);
    try sl.insert(7, 10);
    try sl.insert(10, 5);
    try sl.insert(7, 7);
    try sl.insert(5, 5);
    try sl.insert(10, 10);

    try testing.expectEqual(@as(?i64, 12), try sl.find(1));
    try testing.expectEqual(@as(?i64, 7), try sl.find(7));
    try testing.expectEqual(@as(?i64, 5), try sl.find(5));
    try testing.expectEqual(@as(?i64, 10), try sl.find(10));

    const keys = try sl.keys(testing.allocator);
    defer testing.allocator.free(keys);
    try testing.expectEqualSlices(i64, &[_]i64{ 1, 5, 7, 10 }, keys);
}

test "skip list: delete behavior" {
    var sl = try SkipList.init(testing.allocator, 0.5, 16, 3);
    defer sl.deinit();

    try sl.delete(123); // no-op on empty

    try sl.insert(1, 12);
    try sl.insert(22, 13);
    try sl.insert(24, 14);
    try sl.insert(2, 15);

    try sl.delete(22);
    try sl.delete(2);

    try testing.expectEqual(@as(?i64, null), try sl.find(22));
    try testing.expectEqual(@as(?i64, null), try sl.find(2));
    try testing.expectEqual(@as(?i64, 12), try sl.find(1));
    try testing.expectEqual(@as(?i64, 14), try sl.find(24));
}

test "skip list: invalid init and extreme" {
    try testing.expectError(SkipListError.InvalidProbability, SkipList.init(testing.allocator, 0, 16, 0));
    try testing.expectError(SkipListError.InvalidProbability, SkipList.init(testing.allocator, 1, 16, 0));
    try testing.expectError(SkipListError.InvalidMaxLevel, SkipList.init(testing.allocator, 0.5, 0, 0));

    var sl = try SkipList.init(testing.allocator, 0.5, 24, 99);
    defer sl.deinit();

    for (0..30_000) |i| {
        try sl.insert(@intCast(i), @intCast(i * 2));
    }

    for (0..30_000) |i| {
        try testing.expectEqual(@as(?i64, @intCast(i * 2)), try sl.find(@intCast(i)));
    }

    for (0..30_000) |i| {
        try sl.delete(@intCast(i));
    }

    const keys = try sl.keys(testing.allocator);
    defer testing.allocator.free(keys);
    try testing.expectEqual(@as(usize, 0), keys.len);
}
