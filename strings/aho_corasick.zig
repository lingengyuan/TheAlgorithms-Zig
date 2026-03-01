//! Aho-Corasick Multi-Pattern String Matching - Zig implementation
//! Reference: https://github.com/TheAlgorithms/Python/blob/master/strings/aho_corasick.py

const std = @import("std");
const testing = std.testing;
const Allocator = std.mem.Allocator;

/// One match emitted by Aho-Corasick.
pub const Match = struct {
    pattern_index: usize,
    position: usize,
};

const Node = struct {
    transitions: std.AutoHashMap(u8, usize),
    fail: usize,
    outputs: std.ArrayListUnmanaged(usize),

    fn init(allocator: Allocator) Node {
        return .{
            .transitions = std.AutoHashMap(u8, usize).init(allocator),
            .fail = 0,
            .outputs = .{},
        };
    }

    fn deinit(self: *Node, allocator: Allocator) void {
        self.transitions.deinit();
        self.outputs.deinit(allocator);
    }
};

/// Aho-Corasick automaton.
/// Build time: O(sum(pattern lengths))
/// Query time: O(text length + number of matches)
pub const Automaton = struct {
    const Self = @This();

    allocator: Allocator,
    patterns: []const []const u8,
    nodes: std.ArrayListUnmanaged(Node),

    pub fn init(allocator: Allocator, patterns: []const []const u8) !Self {
        var self = Self{
            .allocator = allocator,
            .patterns = patterns,
            .nodes = .{},
        };
        try self.nodes.append(allocator, Node.init(allocator)); // root
        errdefer self.deinit();

        for (patterns, 0..) |pattern, pattern_index| {
            try self.addPattern(pattern, pattern_index);
        }
        try self.buildFailureLinks();
        return self;
    }

    pub fn deinit(self: *Self) void {
        for (self.nodes.items) |*node| {
            node.deinit(self.allocator);
        }
        self.nodes.deinit(self.allocator);
        self.* = undefined;
    }

    /// Finds all matches of all patterns in `text`.
    /// Caller owns the returned slice.
    pub fn search(self: *const Self, allocator: Allocator, text: []const u8) ![]Match {
        var out = std.ArrayListUnmanaged(Match){};
        defer out.deinit(allocator);

        var state: usize = 0;
        for (text, 0..) |ch, i| {
            while (state != 0 and self.nodes.items[state].transitions.get(ch) == null) {
                state = self.nodes.items[state].fail;
            }
            state = self.nodes.items[state].transitions.get(ch) orelse 0;

            for (self.nodes.items[state].outputs.items) |pattern_index| {
                const pattern = self.patterns[pattern_index];
                if (pattern.len == 0 or pattern.len > i + 1) continue;
                try out.append(allocator, .{
                    .pattern_index = pattern_index,
                    .position = i + 1 - pattern.len,
                });
            }
        }

        return try out.toOwnedSlice(allocator);
    }

    fn addPattern(self: *Self, pattern: []const u8, pattern_index: usize) !void {
        // Empty pattern is intentionally ignored to avoid matching every position.
        if (pattern.len == 0) return;

        var state: usize = 0;
        for (pattern) |ch| {
            if (self.nodes.items[state].transitions.get(ch)) |next_state| {
                state = next_state;
                continue;
            }

            const new_index = self.nodes.items.len;
            try self.nodes.append(self.allocator, Node.init(self.allocator));
            try self.nodes.items[state].transitions.put(ch, new_index);
            state = new_index;
        }

        try self.nodes.items[state].outputs.append(self.allocator, pattern_index);
    }

    fn buildFailureLinks(self: *Self) !void {
        var queue = std.ArrayListUnmanaged(usize){};
        defer queue.deinit(self.allocator);
        var queue_head: usize = 0;

        var root_it = self.nodes.items[0].transitions.iterator();
        while (root_it.next()) |entry| {
            const child = entry.value_ptr.*;
            self.nodes.items[child].fail = 0;
            try queue.append(self.allocator, child);
        }

        while (queue_head < queue.items.len) : (queue_head += 1) {
            const state = queue.items[queue_head];

            var it = self.nodes.items[state].transitions.iterator();
            while (it.next()) |entry| {
                const ch = entry.key_ptr.*;
                const child = entry.value_ptr.*;

                var fail_state = self.nodes.items[state].fail;
                while (fail_state != 0 and self.nodes.items[fail_state].transitions.get(ch) == null) {
                    fail_state = self.nodes.items[fail_state].fail;
                }

                const fallback = self.nodes.items[fail_state].transitions.get(ch) orelse 0;
                self.nodes.items[child].fail = fallback;

                for (self.nodes.items[fallback].outputs.items) |pattern_index| {
                    try self.nodes.items[child].outputs.append(self.allocator, pattern_index);
                }

                try queue.append(self.allocator, child);
            }
        }
    }
};

/// Convenience wrapper for one-shot matching.
pub fn findMatches(
    allocator: Allocator,
    patterns: []const []const u8,
    text: []const u8,
) ![]Match {
    var automaton = try Automaton.init(allocator, patterns);
    defer automaton.deinit();
    return try automaton.search(allocator, text);
}

fn collectPositions(
    allocator: Allocator,
    matches: []const Match,
    pattern_index: usize,
) ![]usize {
    var out = std.ArrayListUnmanaged(usize){};
    defer out.deinit(allocator);
    for (matches) |m| {
        if (m.pattern_index == pattern_index) {
            try out.append(allocator, m.position);
        }
    }
    return try out.toOwnedSlice(allocator);
}

test "aho corasick: example from python reference" {
    const patterns = [_][]const u8{ "what", "hat", "ver", "er" };
    const text = "whatever, err ... , wherever";

    const matches = try findMatches(testing.allocator, &patterns, text);
    defer testing.allocator.free(matches);

    const what_pos = try collectPositions(testing.allocator, matches, 0);
    defer testing.allocator.free(what_pos);
    const hat_pos = try collectPositions(testing.allocator, matches, 1);
    defer testing.allocator.free(hat_pos);
    const ver_pos = try collectPositions(testing.allocator, matches, 2);
    defer testing.allocator.free(ver_pos);
    const er_pos = try collectPositions(testing.allocator, matches, 3);
    defer testing.allocator.free(er_pos);

    try testing.expectEqualSlices(usize, &[_]usize{0}, what_pos);
    try testing.expectEqualSlices(usize, &[_]usize{1}, hat_pos);
    try testing.expectEqualSlices(usize, &[_]usize{ 5, 25 }, ver_pos);
    try testing.expectEqualSlices(usize, &[_]usize{ 6, 10, 22, 26 }, er_pos);
}

test "aho corasick: overlapping patterns" {
    const patterns = [_][]const u8{ "a", "aa", "aaa" };
    const matches = try findMatches(testing.allocator, &patterns, "aaaa");
    defer testing.allocator.free(matches);

    const p0 = try collectPositions(testing.allocator, matches, 0);
    defer testing.allocator.free(p0);
    const p1 = try collectPositions(testing.allocator, matches, 1);
    defer testing.allocator.free(p1);
    const p2 = try collectPositions(testing.allocator, matches, 2);
    defer testing.allocator.free(p2);

    try testing.expectEqualSlices(usize, &[_]usize{ 0, 1, 2, 3 }, p0);
    try testing.expectEqualSlices(usize, &[_]usize{ 0, 1, 2 }, p1);
    try testing.expectEqualSlices(usize, &[_]usize{ 0, 1 }, p2);
}

test "aho corasick: empty text and empty patterns" {
    const patterns = [_][]const u8{ "", "abc", "" };
    const matches = try findMatches(testing.allocator, &patterns, "");
    defer testing.allocator.free(matches);
    try testing.expectEqual(@as(usize, 0), matches.len);
}

test "aho corasick: no match" {
    const patterns = [_][]const u8{ "xyz", "uvw" };
    const matches = try findMatches(testing.allocator, &patterns, "aaaaaaaaaa");
    defer testing.allocator.free(matches);
    try testing.expectEqual(@as(usize, 0), matches.len);
}

test "aho corasick: extreme long repeated input" {
    const patterns = [_][]const u8{ "ab", "bab", "aba" };
    const text = "abababababababababababababababab";
    const matches = try findMatches(testing.allocator, &patterns, text);
    defer testing.allocator.free(matches);

    // Ensure algorithm emits many overlaps correctly and does not miss terminal matches.
    try testing.expect(matches.len > 20);
    try testing.expect(matches[matches.len - 1].position + 1 <= text.len);
}
