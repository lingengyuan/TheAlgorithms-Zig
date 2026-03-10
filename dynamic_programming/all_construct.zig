//! All Construct - Zig implementation
//! Reference: https://github.com/TheAlgorithms/Python/blob/master/dynamic_programming/all_construct.py

const std = @import("std");
const testing = std.testing;
const Allocator = std.mem.Allocator;

pub const AllConstructError = Allocator.Error;
pub const StringCombination = []const []const u8;

/// Returns every ordered construction of `target` using words from `word_bank`.
/// Time complexity: exponential in the output size, Space complexity: O(output size)
pub fn allConstruct(
    allocator: Allocator,
    target: []const u8,
    word_bank: []const []const u8,
) AllConstructError![]StringCombination {
    var path = std.ArrayListUnmanaged([]const u8){};
    defer path.deinit(allocator);

    var results = std.ArrayListUnmanaged(StringCombination){};
    errdefer freeAllConstructResults(allocator, results.items);

    try dfsAllConstruct(allocator, target, word_bank, 0, &path, &results);
    return results.toOwnedSlice(allocator);
}

fn dfsAllConstruct(
    allocator: Allocator,
    target: []const u8,
    word_bank: []const []const u8,
    index: usize,
    path: *std.ArrayListUnmanaged([]const u8),
    results: *std.ArrayListUnmanaged(StringCombination),
) AllConstructError!void {
    if (index == target.len) {
        const combination = try allocator.alloc([]const u8, path.items.len);
        @memcpy(combination, path.items);
        try results.append(allocator, combination);
        return;
    }

    for (word_bank) |word| {
        if (word.len > target.len - index) continue;
        if (!std.mem.eql(u8, target[index .. index + word.len], word)) continue;

        try path.append(allocator, word);
        defer _ = path.pop();
        try dfsAllConstruct(allocator, target, word_bank, index + word.len, path, results);
    }
}

pub fn freeAllConstructResults(allocator: Allocator, combinations: []const StringCombination) void {
    for (combinations) |combination| allocator.free(combination);
    allocator.free(combinations);
}

test "all construct: python samples" {
    const bank = [_][]const u8{ "purp", "p", "ur", "le", "purpl" };
    const combinations = try allConstruct(testing.allocator, "purple", &bank);
    defer freeAllConstructResults(testing.allocator, combinations);

    try testing.expectEqual(@as(usize, 2), combinations.len);
    try testing.expectEqualSlices([]const u8, &[_][]const u8{ "purp", "le" }, combinations[0]);
    try testing.expectEqualSlices([]const u8, &[_][]const u8{ "p", "ur", "p", "le" }, combinations[1]);
}

test "all construct: impossible target" {
    const bank = [_][]const u8{ "bo", "rd", "ate", "ska", "sk", "boar" };
    const combinations = try allConstruct(testing.allocator, "skateboard", &bank);
    defer freeAllConstructResults(testing.allocator, combinations);

    try testing.expectEqual(@as(usize, 0), combinations.len);
}

test "all construct: empty target returns empty combination" {
    const bank = [_][]const u8{ "cat", "dog" };
    const combinations = try allConstruct(testing.allocator, "", &bank);
    defer freeAllConstructResults(testing.allocator, combinations);

    try testing.expectEqual(@as(usize, 1), combinations.len);
    try testing.expectEqual(@as(usize, 0), combinations[0].len);
}

test "all construct: extreme branching count" {
    const bank = [_][]const u8{ "a", "aa", "aaa" };
    const combinations = try allConstruct(testing.allocator, "aaaaaa", &bank);
    defer freeAllConstructResults(testing.allocator, combinations);

    try testing.expectEqual(@as(usize, 24), combinations.len);
}
