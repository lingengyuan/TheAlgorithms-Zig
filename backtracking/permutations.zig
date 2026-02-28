//! All Permutations - Zig implementation
//! Reference: https://github.com/TheAlgorithms/Python/blob/master/backtracking/all_permutations.py

const std = @import("std");
const testing = std.testing;

/// Generates all permutations of `items`. Collected into `result` as owned slices.
/// Caller must free each inner slice and call result.deinit().
pub fn permutations(
    allocator: std.mem.Allocator,
    items: []i32,
    start: usize,
    result: *std.ArrayListUnmanaged([]i32),
) !void {
    if (start == items.len) {
        const copy = try allocator.dupe(i32, items);
        try result.append(allocator, copy);
        return;
    }
    for (start..items.len) |i| {
        std.mem.swap(i32, &items[start], &items[i]);
        try permutations(allocator, items, start + 1, result);
        std.mem.swap(i32, &items[start], &items[i]);
    }
}

test "permutations: [1,2,3]" {
    const alloc = testing.allocator;
    var items = [_]i32{ 1, 2, 3 };
    var result = std.ArrayListUnmanaged([]i32){};
    defer {
        for (result.items) |p| alloc.free(p);
        result.deinit(alloc);
    }

    try permutations(alloc, &items, 0, &result);
    try testing.expectEqual(@as(usize, 6), result.items.len); // 3! = 6

    // Swap-based order: [1,2,3],[1,3,2],[2,1,3],[2,3,1],[3,2,1],[3,1,2]
    try testing.expectEqualSlices(i32, &[_]i32{ 1, 2, 3 }, result.items[0]);
    try testing.expectEqualSlices(i32, &[_]i32{ 3, 1, 2 }, result.items[5]);
}

test "permutations: single element" {
    const alloc = testing.allocator;
    var items = [_]i32{42};
    var result = std.ArrayListUnmanaged([]i32){};
    defer {
        for (result.items) |p| alloc.free(p);
        result.deinit(alloc);
    }
    try permutations(alloc, &items, 0, &result);
    try testing.expectEqual(@as(usize, 1), result.items.len);
    try testing.expectEqualSlices(i32, &[_]i32{42}, result.items[0]);
}

test "permutations: [1,2] gives 2 permutations" {
    const alloc = testing.allocator;
    var items = [_]i32{ 1, 2 };
    var result = std.ArrayListUnmanaged([]i32){};
    defer {
        for (result.items) |p| alloc.free(p);
        result.deinit(alloc);
    }
    try permutations(alloc, &items, 0, &result);
    try testing.expectEqual(@as(usize, 2), result.items.len);
}
