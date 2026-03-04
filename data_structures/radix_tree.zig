//! Radix Tree - Zig implementation
//! Reference: https://github.com/TheAlgorithms/Python/blob/master/data_structures/trie/radix_tree.py

const std = @import("std");
const testing = std.testing;

pub const MatchResult = struct {
    matching: []const u8,
    remaining_prefix: []const u8,
    remaining_word: []const u8,
};

pub const RadixNode = struct {
    nodes: std.AutoHashMap(u8, *RadixNode),
    is_leaf: bool,
    prefix: []u8,
    allocator: std.mem.Allocator,

    pub fn create(allocator: std.mem.Allocator, prefix: []const u8, is_leaf: bool) !*RadixNode {
        const node = try allocator.create(RadixNode);
        errdefer allocator.destroy(node);

        const p = try allocator.dupe(u8, prefix);
        node.* = .{
            .nodes = std.AutoHashMap(u8, *RadixNode).init(allocator),
            .is_leaf = is_leaf,
            .prefix = p,
            .allocator = allocator,
        };
        return node;
    }

    pub fn destroyTree(root: *RadixNode) void {
        const allocator = root.allocator;

        var stack = std.ArrayListUnmanaged(*RadixNode){};
        defer stack.deinit(allocator);
        stack.append(allocator, root) catch return;

        while (stack.items.len > 0) {
            const node = stack.pop().?;
            var it = node.nodes.iterator();
            while (it.next()) |entry| {
                stack.append(allocator, entry.value_ptr.*) catch {};
            }
            node.nodes.deinit();
            allocator.free(node.prefix);
            allocator.destroy(node);
        }
    }

    fn commonPrefixLen(a: []const u8, b: []const u8) usize {
        const n = @min(a.len, b.len);
        var i: usize = 0;
        while (i < n and a[i] == b[i]) : (i += 1) {}
        return i;
    }

    pub fn match(self: *const RadixNode, word: []const u8) MatchResult {
        const x = commonPrefixLen(self.prefix, word);
        return .{
            .matching = self.prefix[0..x],
            .remaining_prefix = self.prefix[x..],
            .remaining_word = word[x..],
        };
    }

    pub fn insertMany(self: *RadixNode, words: []const []const u8) !void {
        for (words) |w| try self.insert(w);
    }

    fn setPrefix(self: *RadixNode, new_prefix: []const u8) !void {
        self.allocator.free(self.prefix);
        self.prefix = try self.allocator.dupe(u8, new_prefix);
    }

    /// Inserts word into radix tree.
    /// Time complexity: O(L), Space complexity: O(L)
    pub fn insert(self: *RadixNode, word: []const u8) !void {
        if (std.mem.eql(u8, self.prefix, word) and !self.is_leaf) {
            self.is_leaf = true;
            return;
        }

        if (word.len == 0) {
            self.is_leaf = true;
            return;
        }

        const first = word[0];
        if (!self.nodes.contains(first)) {
            const child = try RadixNode.create(self.allocator, word, true);
            try self.nodes.put(first, child);
            return;
        }

        var incoming = self.nodes.get(first).?;
        const m = incoming.match(word);

        if (m.remaining_prefix.len == 0) {
            try incoming.insert(m.remaining_word);
            return;
        }

        // Split existing edge with an intermediate node.
        const matching = try self.allocator.dupe(u8, m.matching);
        defer self.allocator.free(matching);
        const old_remaining_prefix = try self.allocator.dupe(u8, m.remaining_prefix);
        defer self.allocator.free(old_remaining_prefix);

        try incoming.setPrefix(old_remaining_prefix);

        const middle = try RadixNode.create(self.allocator, matching, false);
        try middle.nodes.put(old_remaining_prefix[0], incoming);
        try self.nodes.put(matching[0], middle);

        if (m.remaining_word.len == 0) {
            middle.is_leaf = true;
        } else {
            try middle.insert(m.remaining_word);
        }
    }

    /// Returns whether word exists in tree.
    /// Time complexity: O(L), Space complexity: O(L)
    pub fn find(self: *const RadixNode, word: []const u8) bool {
        if (word.len == 0) return false;

        const incoming = self.nodes.get(word[0]) orelse return false;
        const m = incoming.match(word);

        if (m.remaining_prefix.len != 0) return false;
        if (m.remaining_word.len == 0) return incoming.is_leaf;
        return incoming.find(m.remaining_word);
    }

    fn mergeWithOnlyChild(self: *RadixNode) !void {
        if (self.nodes.count() != 1) return;

        var it = self.nodes.iterator();
        const entry = it.next().?;
        const child = entry.value_ptr.*;

        const new_prefix = try std.mem.concat(self.allocator, u8, &[_][]const u8{ self.prefix, child.prefix });
        self.allocator.free(self.prefix);
        self.prefix = new_prefix;

        self.is_leaf = child.is_leaf;

        const child_nodes = child.nodes;
        child.nodes = std.AutoHashMap(u8, *RadixNode).init(self.allocator);

        self.nodes.deinit();
        self.nodes = child_nodes;

        self.allocator.free(child.prefix);
        self.allocator.destroy(child);
    }

    /// Deletes word if present.
    /// Time complexity: O(L), Space complexity: O(L)
    pub fn delete(self: *RadixNode, word: []const u8) bool {
        if (word.len == 0) return false;

        const incoming = self.nodes.get(word[0]) orelse return false;
        const m = incoming.match(word);

        if (m.remaining_prefix.len != 0) return false;

        if (m.remaining_word.len != 0) {
            return incoming.delete(m.remaining_word);
        }

        if (!incoming.is_leaf) return false;

        if (incoming.nodes.count() == 0) {
            _ = self.nodes.remove(word[0]);
            incoming.nodes.deinit();
            self.allocator.free(incoming.prefix);
            self.allocator.destroy(incoming);

            if (self.nodes.count() == 1 and !self.is_leaf) {
                self.mergeWithOnlyChild() catch {};
            }
        } else if (incoming.nodes.count() > 1) {
            incoming.is_leaf = false;
        } else {
            incoming.is_leaf = false;
            incoming.mergeWithOnlyChild() catch {};
        }

        return true;
    }
};

test "radix tree: match example" {
    const node = try RadixNode.create(testing.allocator, "myprefix", false);
    defer RadixNode.destroyTree(node);

    const m = node.match("mystring");
    try testing.expectEqualStrings("my", m.matching);
    try testing.expectEqualStrings("prefix", m.remaining_prefix);
    try testing.expectEqualStrings("string", m.remaining_word);
}

test "radix tree: python test_trie flow" {
    const words = [_][]const u8{ "banana", "bananas", "bandana", "band", "apple", "all", "beast" };

    const root = try RadixNode.create(testing.allocator, "", false);
    defer RadixNode.destroyTree(root);

    try root.insertMany(&words);

    for (words) |w| {
        try testing.expect(root.find(w));
    }

    try testing.expect(!root.find("bandanas"));
    try testing.expect(!root.find("apps"));

    try testing.expect(root.delete("all"));
    try testing.expect(!root.find("all"));

    try testing.expect(root.delete("banana"));
    try testing.expect(!root.find("banana"));
    try testing.expect(root.find("bananas"));
}

test "radix tree: boundary" {
    const root = try RadixNode.create(testing.allocator, "", false);
    defer RadixNode.destroyTree(root);

    try testing.expect(!root.find("a"));
    try testing.expect(!root.delete("a"));

    try root.insert("abc");
    try testing.expect(root.find("abc"));
    try testing.expect(!root.find("ab"));
    try testing.expect(root.delete("abc"));
    try testing.expect(!root.find("abc"));
}

test "radix tree: extreme bulk insert find delete" {
    const root = try RadixNode.create(testing.allocator, "", false);
    defer RadixNode.destroyTree(root);

    const n: usize = 15_000;

    var i: usize = 0;
    while (i < n) : (i += 1) {
        const w = try std.fmt.allocPrint(testing.allocator, "word-{d}", .{i});
        defer testing.allocator.free(w);
        try root.insert(w);
    }

    i = 0;
    while (i < n) : (i += 997) {
        const w = try std.fmt.allocPrint(testing.allocator, "word-{d}", .{i});
        defer testing.allocator.free(w);
        try testing.expect(root.find(w));
    }

    i = 0;
    while (i < n) : (i += 2) {
        const w = try std.fmt.allocPrint(testing.allocator, "word-{d}", .{i});
        defer testing.allocator.free(w);
        try testing.expect(root.delete(w));
    }

    i = 0;
    while (i < n) : (i += 997) {
        const w = try std.fmt.allocPrint(testing.allocator, "word-{d}", .{i});
        defer testing.allocator.free(w);
        if (i % 2 == 0) {
            try testing.expect(!root.find(w));
        } else {
            try testing.expect(root.find(w));
        }
    }
}
