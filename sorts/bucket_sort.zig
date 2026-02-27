//! Bucket Sort - Zig implementation
//! Reference: https://github.com/TheAlgorithms/Python/blob/master/sorts/bucket_sort.py

const std = @import("std");
const testing = std.testing;

/// Bucket sort for i32 slices. Returns a newly allocated sorted slice.
/// Uses sqrt(n) buckets and insertion sort inside each bucket.
/// Caller owns the returned memory.
/// Time complexity: O(n + k) average, O(n²) worst, Space complexity: O(n + k)
pub fn bucketSort(allocator: std.mem.Allocator, items: []const i32) ![]i32 {
    if (items.len == 0) {
        return try allocator.alloc(i32, 0);
    }
    if (items.len == 1) {
        const output = try allocator.alloc(i32, 1);
        output[0] = items[0];
        return output;
    }

    var min_val: i32 = items[0];
    var max_val: i32 = items[0];
    for (items[1..]) |v| {
        if (v < min_val) min_val = v;
        if (v > max_val) max_val = v;
    }

    if (min_val == max_val) {
        const output = try allocator.alloc(i32, items.len);
        for (items, 0..) |v, i| output[i] = v;
        return output;
    }

    const bucket_count = @max(@as(usize, 1), std.math.sqrt(items.len));
    const counts = try allocator.alloc(usize, bucket_count);
    defer allocator.free(counts);
    @memset(counts, 0);

    for (items) |v| {
        const idx = bucketIndex(v, min_val, max_val, bucket_count);
        counts[idx] += 1;
    }

    const offsets = try allocator.alloc(usize, bucket_count + 1);
    defer allocator.free(offsets);
    offsets[0] = 0;
    for (0..bucket_count) |i| {
        offsets[i + 1] = offsets[i] + counts[i];
    }

    const next = try allocator.alloc(usize, bucket_count);
    defer allocator.free(next);
    for (0..bucket_count) |i| next[i] = offsets[i];

    const output = try allocator.alloc(i32, items.len);
    for (items) |v| {
        const idx = bucketIndex(v, min_val, max_val, bucket_count);
        const pos = next[idx];
        output[pos] = v;
        next[idx] += 1;
    }

    for (0..bucket_count) |i| {
        insertionSort(output[offsets[i]..offsets[i + 1]]);
    }

    return output;
}

fn bucketIndex(value: i32, min_val: i32, max_val: i32, bucket_count: usize) usize {
    const numerator = (@as(i64, value) - @as(i64, min_val)) * @as(i64, @intCast(bucket_count));
    const denominator = (@as(i64, max_val) - @as(i64, min_val)) + 1;
    var idx: usize = @intCast(@divTrunc(numerator, denominator));
    if (idx >= bucket_count) idx = bucket_count - 1;
    return idx;
}

fn insertionSort(arr: []i32) void {
    if (arr.len <= 1) return;

    for (1..arr.len) |i| {
        const key = arr[i];
        var j = i;
        while (j > 0 and arr[j - 1] > key) {
            arr[j] = arr[j - 1];
            j -= 1;
        }
        arr[j] = key;
    }
}

test "bucket sort: basic case" {
    const alloc = testing.allocator;
    const sorted = try bucketSort(alloc, &[_]i32{ 42, 32, 33, 52, 37, 47, 51 });
    defer alloc.free(sorted);
    try testing.expectEqualSlices(i32, &[_]i32{ 32, 33, 37, 42, 47, 51, 52 }, sorted);
}

test "bucket sort: handles negatives" {
    const alloc = testing.allocator;
    const sorted = try bucketSort(alloc, &[_]i32{ -5, 10, 0, -1, 7, -12 });
    defer alloc.free(sorted);
    try testing.expectEqualSlices(i32, &[_]i32{ -12, -5, -1, 0, 7, 10 }, sorted);
}

test "bucket sort: empty array" {
    const alloc = testing.allocator;
    const sorted = try bucketSort(alloc, &[_]i32{});
    defer alloc.free(sorted);
    try testing.expectEqual(@as(usize, 0), sorted.len);
}

test "bucket sort: single element" {
    const alloc = testing.allocator;
    const sorted = try bucketSort(alloc, &[_]i32{7});
    defer alloc.free(sorted);
    try testing.expectEqualSlices(i32, &[_]i32{7}, sorted);
}

test "bucket sort: all equal" {
    const alloc = testing.allocator;
    const sorted = try bucketSort(alloc, &[_]i32{ 5, 5, 5, 5 });
    defer alloc.free(sorted);
    try testing.expectEqualSlices(i32, &[_]i32{ 5, 5, 5, 5 }, sorted);
}

test "bucket sort: already sorted" {
    const alloc = testing.allocator;
    const sorted = try bucketSort(alloc, &[_]i32{ -3, -1, 0, 2, 8, 9 });
    defer alloc.free(sorted);
    try testing.expectEqualSlices(i32, &[_]i32{ -3, -1, 0, 2, 8, 9 }, sorted);
}
