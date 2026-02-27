//! Merge Sort - Zig implementation
//! Reference: https://github.com/TheAlgorithms/Python/blob/master/sorts/merge_sort.py

const std = @import("std");
const testing = std.testing;

/// Merge sort: returns a newly allocated sorted slice. Caller owns the returned memory.
/// Time complexity: O(n log n), Space complexity: O(n)
pub fn mergeSort(comptime T: type, allocator: std.mem.Allocator, items: []const T) ![]T {
    const len = items.len;
    if (len <= 1) {
        const result = try allocator.alloc(T, len);
        if (len == 1) result[0] = items[0];
        return result;
    }

    const mid = len / 2;
    const left = try mergeSort(T, allocator, items[0..mid]);
    defer allocator.free(left);
    const right = try mergeSort(T, allocator, items[mid..]);
    defer allocator.free(right);

    return merge(T, allocator, left, right);
}

fn merge(comptime T: type, allocator: std.mem.Allocator, left: []const T, right: []const T) ![]T {
    const result = try allocator.alloc(T, left.len + right.len);
    var i: usize = 0;
    var l: usize = 0;
    var r: usize = 0;

    while (l < left.len and r < right.len) {
        if (left[l] <= right[r]) {
            result[i] = left[l];
            l += 1;
        } else {
            result[i] = right[r];
            r += 1;
        }
        i += 1;
    }
    while (l < left.len) {
        result[i] = left[l];
        l += 1;
        i += 1;
    }
    while (r < right.len) {
        result[i] = right[r];
        r += 1;
        i += 1;
    }
    return result;
}

// ===== Tests =====

test "merge sort: basic case" {
    const allocator = testing.allocator;
    const input = [_]i32{ 0, 5, 3, 2, 2 };
    const sorted = try mergeSort(i32, allocator, &input);
    defer allocator.free(sorted);
    try testing.expectEqualSlices(i32, &[_]i32{ 0, 2, 2, 3, 5 }, sorted);
}

test "merge sort: empty array" {
    const allocator = testing.allocator;
    const input = [_]i32{};
    const sorted = try mergeSort(i32, allocator, &input);
    defer allocator.free(sorted);
    try testing.expectEqual(@as(usize, 0), sorted.len);
}

test "merge sort: all negative" {
    const allocator = testing.allocator;
    const input = [_]i32{ -2, -5, -45 };
    const sorted = try mergeSort(i32, allocator, &input);
    defer allocator.free(sorted);
    try testing.expectEqualSlices(i32, &[_]i32{ -45, -5, -2 }, sorted);
}

test "merge sort: already sorted" {
    const allocator = testing.allocator;
    const input = [_]i32{ 1, 2, 3, 4, 5 };
    const sorted = try mergeSort(i32, allocator, &input);
    defer allocator.free(sorted);
    try testing.expectEqualSlices(i32, &[_]i32{ 1, 2, 3, 4, 5 }, sorted);
}

test "merge sort: reverse sorted" {
    const allocator = testing.allocator;
    const input = [_]i32{ 5, 4, 3, 2, 1 };
    const sorted = try mergeSort(i32, allocator, &input);
    defer allocator.free(sorted);
    try testing.expectEqualSlices(i32, &[_]i32{ 1, 2, 3, 4, 5 }, sorted);
}

test "merge sort: single element" {
    const allocator = testing.allocator;
    const input = [_]i32{42};
    const sorted = try mergeSort(i32, allocator, &input);
    defer allocator.free(sorted);
    try testing.expectEqualSlices(i32, &[_]i32{42}, sorted);
}

test "merge sort: large array" {
    const allocator = testing.allocator;
    const input = [_]i32{ 38, 27, 43, 3, 9, 82, 10 };
    const sorted = try mergeSort(i32, allocator, &input);
    defer allocator.free(sorted);
    try testing.expectEqualSlices(i32, &[_]i32{ 3, 9, 10, 27, 38, 43, 82 }, sorted);
}
