//! Trie (Prefix Tree) - Zig implementation
//! Reference: https://github.com/TheAlgorithms/Python/tree/master/data_structures/trie

const std = @import("std");
const testing = std.testing;
const Allocator = std.mem.Allocator;

/// Trie for lowercase ASCII words (`a`-`z`).
/// Insert/search/prefix/delete time complexity: O(L), space: O(total characters)
pub const Trie = struct {
    const Self = @This();

    pub const Node = struct {
        children: [26]?*Node,
        is_end: bool,

        fn init() Node {
            return .{
                .children = [_]?*Node{null} ** 26,
                .is_end = false,
            };
        }
    };

    allocator: Allocator,
    root: *Node,

    pub fn init(allocator: Allocator) !Self {
        const root = try allocator.create(Node);
        root.* = Node.init();
        return .{ .allocator = allocator, .root = root };
    }

    pub fn deinit(self: *Self) void {
        freeNode(self.allocator, self.root);
        self.* = undefined;
    }

    fn freeNode(allocator: Allocator, node: *Node) void {
        for (node.children) |child| {
            if (child) |c| freeNode(allocator, c);
        }
        allocator.destroy(node);
    }

    pub fn insert(self: *Self, word: []const u8) !void {
        var cur = self.root;
        for (word) |ch| {
            const idx = charIndex(ch) orelse return error.InvalidCharacter;
            if (cur.children[idx] == null) {
                const child = try self.allocator.create(Node);
                child.* = Node.init();
                cur.children[idx] = child;
            }
            cur = cur.children[idx].?;
        }
        cur.is_end = true;
    }

    pub fn contains(self: *const Self, word: []const u8) bool {
        const node = self.findNode(word) orelse return false;
        return node.is_end;
    }

    pub fn startsWith(self: *const Self, prefix: []const u8) bool {
        return self.findNode(prefix) != null;
    }

    pub fn remove(self: *Self, word: []const u8) bool {
        var removed = false;
        _ = removeRec(self.allocator, self.root, word, 0, &removed);
        return removed;
    }

    fn removeRec(allocator: Allocator, node: *Node, word: []const u8, depth: usize, removed: *bool) bool {
        if (depth == word.len) {
            if (!node.is_end) return false;
            node.is_end = false;
            removed.* = true;
            return !hasAnyChild(node);
        }

        const idx = charIndex(word[depth]) orelse return false;
        const child = node.children[idx] orelse return false;
        const should_delete_child = removeRec(allocator, child, word, depth + 1, removed);
        if (should_delete_child) {
            allocator.destroy(child);
            node.children[idx] = null;
        }

        return !node.is_end and !hasAnyChild(node);
    }

    fn findNode(self: *const Self, text: []const u8) ?*Node {
        var cur = self.root;
        for (text) |ch| {
            const idx = charIndex(ch) orelse return null;
            cur = cur.children[idx] orelse return null;
        }
        return cur;
    }

    fn hasAnyChild(node: *const Node) bool {
        for (node.children) |child| {
            if (child != null) return true;
        }
        return false;
    }

    fn charIndex(ch: u8) ?usize {
        if (ch < 'a' or ch > 'z') return null;
        return @as(usize, ch - 'a');
    }
};

test "trie: insert and contains" {
    var trie = try Trie.init(testing.allocator);
    defer trie.deinit();

    try trie.insert("apple");
    try trie.insert("app");
    try trie.insert("banana");

    try testing.expect(trie.contains("apple"));
    try testing.expect(trie.contains("app"));
    try testing.expect(trie.contains("banana"));
    try testing.expect(!trie.contains("appl"));
    try testing.expect(!trie.contains("band"));
}

test "trie: starts with" {
    var trie = try Trie.init(testing.allocator);
    defer trie.deinit();

    try trie.insert("algorithm");
    try trie.insert("algebra");

    try testing.expect(trie.startsWith("alg"));
    try testing.expect(trie.startsWith("al"));
    try testing.expect(!trie.startsWith("ax"));
}

test "trie: remove word" {
    var trie = try Trie.init(testing.allocator);
    defer trie.deinit();

    try trie.insert("app");
    try trie.insert("apple");
    try trie.insert("apply");

    try testing.expect(trie.remove("apple"));
    try testing.expect(!trie.contains("apple"));
    try testing.expect(trie.contains("app"));
    try testing.expect(trie.contains("apply"));
    try testing.expect(!trie.remove("apple"));
}

test "trie: invalid character" {
    var trie = try Trie.init(testing.allocator);
    defer trie.deinit();

    try testing.expectError(error.InvalidCharacter, trie.insert("A"));
    try testing.expectError(error.InvalidCharacter, trie.insert("abc1"));
}
