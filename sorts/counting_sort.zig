//! Counting Sort - Zig implementation
//! Reference: https://github.com/TheAlgorithms/Python/blob/master/sorts/counting_sort.py

const std = @import("std");
const testing = std.testing;
const max_counting_range: u64 = 10_000_000;

/// Counting sort for i32 slices. Returns a newly allocated sorted slice.
/// Caller owns the returned memory.
/// Time complexity: O(n + k), Space complexity: O(n + k) where k = max - min
pub fn countingSort(allocator: std.mem.Allocator, items: []const i32) ![]i32 {
    if (items.len == 0) {
        return try allocator.alloc(i32, 0);
    }

    // Find min and max
    var min_val: i32 = items[0];
    var max_val: i32 = items[0];
    for (items[1..]) |v| {
        if (v < min_val) min_val = v;
        if (v > max_val) max_val = v;
    }

    // Build count array
    const range_i64: i64 = @as(i64, max_val) - @as(i64, min_val) + 1;
    if (range_i64 <= 0) return error.RangeTooLarge;
    const range_u64: u64 = @intCast(range_i64);
    if (range_u64 > std.math.maxInt(usize)) return error.RangeTooLarge;
    if (range_u64 > max_counting_range) return error.RangeTooLarge;
    const range: usize = @intCast(range_u64);
    const counts = try allocator.alloc(usize, range);
    defer allocator.free(counts);
    @memset(counts, 0);

    for (items) |v| {
        const idx: usize = @intCast(@as(i64, v) - @as(i64, min_val));
        counts[idx] += 1;
    }

    // Prefix sum
    for (1..range) |i| {
        counts[i] += counts[i - 1];
    }

    // Build output (stable: iterate in reverse)
    const output = try allocator.alloc(i32, items.len);
    var i: usize = items.len;
    while (i > 0) {
        i -= 1;
        const idx: usize = @intCast(@as(i64, items[i]) - @as(i64, min_val));
        counts[idx] -= 1;
        output[counts[idx]] = items[i];
    }

    return output;
}

test "counting sort: basic case" {
    const alloc = testing.allocator;
    const sorted = try countingSort(alloc, &[_]i32{ 0, 5, 3, 2, 2 });
    defer alloc.free(sorted);
    try testing.expectEqualSlices(i32, &[_]i32{ 0, 2, 2, 3, 5 }, sorted);
}

test "counting sort: empty array" {
    const alloc = testing.allocator;
    const sorted = try countingSort(alloc, &[_]i32{});
    defer alloc.free(sorted);
    try testing.expectEqual(@as(usize, 0), sorted.len);
}

test "counting sort: all negative" {
    const alloc = testing.allocator;
    const sorted = try countingSort(alloc, &[_]i32{ -2, -5, -45 });
    defer alloc.free(sorted);
    try testing.expectEqualSlices(i32, &[_]i32{ -45, -5, -2 }, sorted);
}

test "counting sort: single element" {
    const alloc = testing.allocator;
    const sorted = try countingSort(alloc, &[_]i32{42});
    defer alloc.free(sorted);
    try testing.expectEqualSlices(i32, &[_]i32{42}, sorted);
}

test "counting sort: already sorted" {
    const alloc = testing.allocator;
    const sorted = try countingSort(alloc, &[_]i32{ 1, 2, 3, 4, 5 });
    defer alloc.free(sorted);
    try testing.expectEqualSlices(i32, &[_]i32{ 1, 2, 3, 4, 5 }, sorted);
}

test "counting sort: huge range returns error" {
    const alloc = testing.allocator;
    try testing.expectError(
        error.RangeTooLarge,
        countingSort(alloc, &[_]i32{ std.math.minInt(i32), std.math.maxInt(i32) }),
    );
}
