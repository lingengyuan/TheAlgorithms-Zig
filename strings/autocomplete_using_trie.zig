//! Autocomplete Using Trie - Zig implementation
//! Reference: https://github.com/TheAlgorithms/Python/blob/master/strings/autocomplete_using_trie.py

const std = @import("std");
const testing = std.testing;
const Allocator = std.mem.Allocator;

pub const StringList = []const []u8;

const Node = struct {
    is_end: bool = false,
    children: std.AutoHashMap(u8, *Node),

    fn init(allocator: Allocator) Node {
        return .{ .children = std.AutoHashMap(u8, *Node).init(allocator) };
    }

    fn deinit(self: *Node, allocator: Allocator) void {
        var it = self.children.iterator();
        while (it.next()) |entry| {
            entry.value_ptr.*.deinit(allocator);
            allocator.destroy(entry.value_ptr.*);
        }
        self.children.deinit();
    }
};

pub const Trie = struct {
    allocator: Allocator,
    root: *Node,

    pub fn init(allocator: Allocator) !Trie {
        const root = try allocator.create(Node);
        root.* = Node.init(allocator);
        return .{ .allocator = allocator, .root = root };
    }

    pub fn deinit(self: *Trie) void {
        self.root.deinit(self.allocator);
        self.allocator.destroy(self.root);
    }

    pub fn insertWord(self: *Trie, text: []const u8) !void {
        var node = self.root;
        for (text) |char| {
            const entry = try node.children.getOrPut(char);
            if (!entry.found_existing) {
                entry.value_ptr.* = try self.allocator.create(Node);
                entry.value_ptr.*.* = Node.init(self.allocator);
            }
            node = entry.value_ptr.*;
        }
        node.is_end = true;
    }

    pub fn findWord(self: *Trie, allocator: Allocator, prefix: []const u8) !StringList {
        var node = self.root;
        for (prefix) |char| {
            node = node.children.get(char) orelse return allocator.alloc([]u8, 0);
        }

        var results = std.ArrayListUnmanaged([]u8){};
        errdefer freeStrings(allocator, results.items);
        var buffer = std.ArrayListUnmanaged(u8){};
        defer buffer.deinit(allocator);
        try collectSuffixes(node, allocator, &buffer, &results);
        return results.toOwnedSlice(allocator);
    }

    fn collectSuffixes(node: *Node, allocator: Allocator, buffer: *std.ArrayListUnmanaged(u8), out: *std.ArrayListUnmanaged([]u8)) !void {
        if (node.is_end) {
            try buffer.append(allocator, ' ');
            defer _ = buffer.pop();
            try out.append(allocator, try allocator.dupe(u8, buffer.items));
        }

        var keys = std.ArrayListUnmanaged(u8){};
        defer keys.deinit(allocator);
        var it = node.children.iterator();
        while (it.next()) |entry| try keys.append(allocator, entry.key_ptr.*);
        std.sort.heap(u8, keys.items, {}, std.sort.asc(u8));

        for (keys.items) |char| {
            try buffer.append(allocator, char);
            defer _ = buffer.pop();
            try collectSuffixes(node.children.get(char).?, allocator, buffer, out);
        }
    }
};

pub fn autocompleteUsingTrie(allocator: Allocator, trie: *Trie, prefix: []const u8) !StringList {
    const suffixes = try trie.findWord(allocator, prefix);
    errdefer freeStrings(allocator, suffixes);

    var results = std.ArrayListUnmanaged([]u8){};
    errdefer freeStrings(allocator, results.items);
    for (suffixes) |suffix| {
        try results.append(allocator, try std.fmt.allocPrint(allocator, "{s}{s}", .{ prefix, suffix }));
    }
    freeStrings(allocator, suffixes);
    return results.toOwnedSlice(allocator);
}

pub fn freeStrings(allocator: Allocator, items: StringList) void {
    for (items) |item| allocator.free(item);
    allocator.free(items);
}

test "autocomplete using trie: python sample" {
    var trie = try Trie.init(testing.allocator);
    defer trie.deinit();
    const words = [_][]const u8{ "depart", "detergent", "daring", "dog", "deer", "deal" };
    for (words) |word| try trie.insertWord(word);

    const matches = try autocompleteUsingTrie(testing.allocator, &trie, "de");
    defer freeStrings(testing.allocator, matches);
    try testing.expectEqual(@as(usize, 4), matches.len);
    try testing.expectEqualStrings("deal ", matches[0]);
    try testing.expectEqualStrings("deer ", matches[1]);
    try testing.expectEqualStrings("depart ", matches[2]);
    try testing.expectEqualStrings("detergent ", matches[3]);
}
