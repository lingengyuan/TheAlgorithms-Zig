//! All Combinations (choose k from 1..n) - Zig implementation
//! Reference: https://github.com/TheAlgorithms/Python/blob/master/backtracking/all_combinations.py

const std = @import("std");
const testing = std.testing;

fn combine(
    allocator: std.mem.Allocator,
    n: usize,
    k: usize,
    start: usize,
    current: *std.ArrayListUnmanaged(usize),
    result: *std.ArrayListUnmanaged([]usize),
) !void {
    if (current.items.len == k) {
        const copy = try allocator.dupe(usize, current.items);
        try result.append(allocator, copy);
        return;
    }
    const remaining = k - current.items.len;
    const limit = n - remaining + 1;
    for (start..limit + 1) |i| {
        try current.append(allocator, i);
        try combine(allocator, n, k, i + 1, current, result);
        _ = current.pop();
    }
}

/// Generates all combinations of k numbers chosen from 1..=n.
/// Caller must free each inner slice and call result.deinit().
pub fn combinations(
    allocator: std.mem.Allocator,
    n: usize,
    k: usize,
    result: *std.ArrayListUnmanaged([]usize),
) !void {
    var current = std.ArrayListUnmanaged(usize){};
    defer current.deinit(allocator);
    try combine(allocator, n, k, 1, &current, result);
}

test "combinations: C(4,2)" {
    const alloc = testing.allocator;
    var result = std.ArrayListUnmanaged([]usize){};
    defer {
        for (result.items) |c| alloc.free(c);
        result.deinit(alloc);
    }
    try combinations(alloc, 4, 2, &result);
    try testing.expectEqual(@as(usize, 6), result.items.len);
    try testing.expectEqualSlices(usize, &[_]usize{ 1, 2 }, result.items[0]);
    try testing.expectEqualSlices(usize, &[_]usize{ 3, 4 }, result.items[5]);
}

test "combinations: C(3,3)" {
    const alloc = testing.allocator;
    var result = std.ArrayListUnmanaged([]usize){};
    defer {
        for (result.items) |c| alloc.free(c);
        result.deinit(alloc);
    }
    try combinations(alloc, 3, 3, &result);
    try testing.expectEqual(@as(usize, 1), result.items.len);
    try testing.expectEqualSlices(usize, &[_]usize{ 1, 2, 3 }, result.items[0]);
}

test "combinations: C(3,1)" {
    const alloc = testing.allocator;
    var result = std.ArrayListUnmanaged([]usize){};
    defer {
        for (result.items) |c| alloc.free(c);
        result.deinit(alloc);
    }
    try combinations(alloc, 3, 1, &result);
    try testing.expectEqual(@as(usize, 3), result.items.len);
}
