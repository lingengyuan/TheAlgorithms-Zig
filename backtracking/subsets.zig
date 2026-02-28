//! All Subsets (Power Set) - Zig implementation
//! Reference: https://github.com/TheAlgorithms/Python/blob/master/backtracking/all_subsequences.py

const std = @import("std");
const testing = std.testing;

fn subsetHelper(
    allocator: std.mem.Allocator,
    items: []const i32,
    index: usize,
    current: *std.ArrayListUnmanaged(i32),
    result: *std.ArrayListUnmanaged([]i32),
) !void {
    const copy = try allocator.dupe(i32, current.items);
    try result.append(allocator, copy);

    for (index..items.len) |i| {
        try current.append(allocator, items[i]);
        try subsetHelper(allocator, items, i + 1, current, result);
        _ = current.pop();
    }
}

/// Generates all subsets (power set) of items. 2^n subsets total.
/// Caller must free each inner slice and call result.deinit().
pub fn allSubsets(
    allocator: std.mem.Allocator,
    items: []const i32,
    result: *std.ArrayListUnmanaged([]i32),
) !void {
    var current = std.ArrayListUnmanaged(i32){};
    defer current.deinit(allocator);
    try subsetHelper(allocator, items, 0, &current, result);
}

test "subsets: [1,2,3] has 8 subsets" {
    const alloc = testing.allocator;
    var result = std.ArrayListUnmanaged([]i32){};
    defer {
        for (result.items) |s| alloc.free(s);
        result.deinit(alloc);
    }
    try allSubsets(alloc, &[_]i32{ 1, 2, 3 }, &result);
    try testing.expectEqual(@as(usize, 8), result.items.len); // 2^3 = 8
    // DFS order: [], [1], [1,2], [1,2,3], [1,3], [2], [2,3], [3]
    try testing.expectEqual(@as(usize, 0), result.items[0].len); // empty set
    try testing.expectEqualSlices(i32, &[_]i32{ 1, 2, 3 }, result.items[3]);
    try testing.expectEqualSlices(i32, &[_]i32{3}, result.items[7]);
}

test "subsets: empty input has 1 subset (empty set)" {
    const alloc = testing.allocator;
    var result = std.ArrayListUnmanaged([]i32){};
    defer {
        for (result.items) |s| alloc.free(s);
        result.deinit(alloc);
    }
    try allSubsets(alloc, &[_]i32{}, &result);
    try testing.expectEqual(@as(usize, 1), result.items.len);
    try testing.expectEqual(@as(usize, 0), result.items[0].len);
}

test "subsets: [1] has 2 subsets" {
    const alloc = testing.allocator;
    var result = std.ArrayListUnmanaged([]i32){};
    defer {
        for (result.items) |s| alloc.free(s);
        result.deinit(alloc);
    }
    try allSubsets(alloc, &[_]i32{1}, &result);
    try testing.expectEqual(@as(usize, 2), result.items.len);
}
