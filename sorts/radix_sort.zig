//! Radix Sort - Zig implementation
//! Reference: https://github.com/TheAlgorithms/Python/blob/master/sorts/radix_sort.py

const std = @import("std");
const testing = std.testing;

/// Radix sort for i32 slices. Returns a newly allocated sorted slice.
/// Supports negative values by sorting absolute values separately.
/// Caller owns the returned memory.
/// Time complexity: O(d * (n + b)), Space complexity: O(n + b), where b = 10
pub fn radixSort(allocator: std.mem.Allocator, items: []const i32) ![]i32 {
    if (items.len == 0) {
        return try allocator.alloc(i32, 0);
    }

    var neg_count: usize = 0;
    for (items) |v| {
        if (v < 0) neg_count += 1;
    }
    const pos_count = items.len - neg_count;

    const negatives = try allocator.alloc(u32, neg_count);
    defer allocator.free(negatives);
    const positives = try allocator.alloc(u32, pos_count);
    defer allocator.free(positives);

    var ni: usize = 0;
    var pi: usize = 0;
    for (items) |v| {
        if (v < 0) {
            negatives[ni] = @intCast(-@as(i64, v));
            ni += 1;
        } else {
            positives[pi] = @intCast(v);
            pi += 1;
        }
    }

    try radixSortUnsigned(allocator, negatives);
    try radixSortUnsigned(allocator, positives);

    const output = try allocator.alloc(i32, items.len);

    var out_idx: usize = 0;
    var i = negatives.len;
    while (i > 0) {
        i -= 1;
        const value_i64 = -@as(i64, negatives[i]);
        output[out_idx] = @intCast(value_i64);
        out_idx += 1;
    }

    for (positives) |v| {
        output[out_idx] = @intCast(v);
        out_idx += 1;
    }

    return output;
}

fn radixSortUnsigned(allocator: std.mem.Allocator, arr: []u32) !void {
    if (arr.len <= 1) return;

    const output = try allocator.alloc(u32, arr.len);
    defer allocator.free(output);

    var max_val: u32 = arr[0];
    for (arr[1..]) |v| {
        if (v > max_val) max_val = v;
    }

    var exp: u32 = 1;
    while (true) {
        var count = [_]usize{0} ** 10;

        for (arr) |value| {
            const digit: usize = @intCast((value / exp) % 10);
            count[digit] += 1;
        }

        for (1..10) |idx| {
            count[idx] += count[idx - 1];
        }

        var idx = arr.len;
        while (idx > 0) {
            idx -= 1;
            const digit: usize = @intCast((arr[idx] / exp) % 10);
            count[digit] -= 1;
            output[count[digit]] = arr[idx];
        }

        for (0..arr.len) |j| {
            arr[j] = output[j];
        }

        if (exp > max_val / 10) break;
        exp *= 10;
    }
}

test "radix sort: basic case" {
    const alloc = testing.allocator;
    const sorted = try radixSort(alloc, &[_]i32{ 170, 45, 75, 90, 802, 24, 2, 66 });
    defer alloc.free(sorted);
    try testing.expectEqualSlices(i32, &[_]i32{ 2, 24, 45, 66, 75, 90, 170, 802 }, sorted);
}

test "radix sort: handles negatives" {
    const alloc = testing.allocator;
    const sorted = try radixSort(alloc, &[_]i32{ -5, 10, 0, -1, 7, -12 });
    defer alloc.free(sorted);
    try testing.expectEqualSlices(i32, &[_]i32{ -12, -5, -1, 0, 7, 10 }, sorted);
}

test "radix sort: empty array" {
    const alloc = testing.allocator;
    const sorted = try radixSort(alloc, &[_]i32{});
    defer alloc.free(sorted);
    try testing.expectEqual(@as(usize, 0), sorted.len);
}

test "radix sort: single element" {
    const alloc = testing.allocator;
    const sorted = try radixSort(alloc, &[_]i32{42});
    defer alloc.free(sorted);
    try testing.expectEqualSlices(i32, &[_]i32{42}, sorted);
}

test "radix sort: already sorted" {
    const alloc = testing.allocator;
    const sorted = try radixSort(alloc, &[_]i32{ -3, -1, 0, 2, 8, 9 });
    defer alloc.free(sorted);
    try testing.expectEqualSlices(i32, &[_]i32{ -3, -1, 0, 2, 8, 9 }, sorted);
}

test "radix sort: includes i32 boundaries" {
    const alloc = testing.allocator;
    const sorted = try radixSort(alloc, &[_]i32{ 0, -2147483648, 2147483647, -1 });
    defer alloc.free(sorted);
    try testing.expectEqualSlices(i32, &[_]i32{ -2147483648, -1, 0, 2147483647 }, sorted);
}
