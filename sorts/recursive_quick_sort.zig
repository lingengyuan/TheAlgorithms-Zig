//! Recursive Quick Sort - Zig implementation
//! Reference: https://github.com/TheAlgorithms/Python/blob/master/sorts/recursive_quick_sort.py

const std = @import("std");
const testing = std.testing;

/// Returns a newly allocated sorted copy of input data.
/// Caller owns returned slice.
/// Time complexity: O(n log n) average, O(n²) worst
/// Space complexity: O(n) auxiliary + recursion
pub fn recursiveQuickSort(comptime T: type, allocator: std.mem.Allocator, data: []const T) ![]T {
    if (data.len <= 1) {
        const out = try allocator.alloc(T, data.len);
        @memcpy(out, data);
        return out;
    }

    const pivot = data[0];

    var lower_or_equal = std.ArrayListUnmanaged(T){};
    defer lower_or_equal.deinit(allocator);
    var greater = std.ArrayListUnmanaged(T){};
    defer greater.deinit(allocator);

    for (data[1..]) |e| {
        if (e <= pivot) {
            try lower_or_equal.append(allocator, e);
        } else {
            try greater.append(allocator, e);
        }
    }

    const left = try recursiveQuickSort(T, allocator, lower_or_equal.items);
    defer allocator.free(left);
    const right = try recursiveQuickSort(T, allocator, greater.items);
    defer allocator.free(right);

    const out = try allocator.alloc(T, data.len);
    @memcpy(out[0..left.len], left);
    out[left.len] = pivot;
    @memcpy(out[left.len + 1 ..], right);
    return out;
}

fn expectSortedAscending(comptime T: type, arr: []const T) !void {
    if (arr.len <= 1) return;
    for (1..arr.len) |i| {
        try testing.expect(arr[i - 1] <= arr[i]);
    }
}

test "recursive quick sort: python reference examples" {
    const alloc = testing.allocator;

    const a1 = [_]i32{ 2, 1, 0 };
    const r1 = try recursiveQuickSort(i32, alloc, &a1);
    defer alloc.free(r1);
    try testing.expectEqualSlices(i32, &[_]i32{ 0, 1, 2 }, r1);

    const a2 = [_]f64{ 2.2, 1.1, 0.0 };
    const r2 = try recursiveQuickSort(f64, alloc, &a2);
    defer alloc.free(r2);
    try testing.expectEqualSlices(f64, &[_]f64{ 0.0, 1.1, 2.2 }, r2);

    const word = "quick_sort";
    const r3 = try recursiveQuickSort(u8, alloc, word);
    defer alloc.free(r3);
    try testing.expectEqualStrings("_cikoqrstu", r3);
}

test "recursive quick sort: edge cases" {
    const alloc = testing.allocator;

    const empty = [_]i32{};
    const r1 = try recursiveQuickSort(i32, alloc, &empty);
    defer alloc.free(r1);
    try testing.expectEqual(@as(usize, 0), r1.len);

    const one = [_]i32{9};
    const r2 = try recursiveQuickSort(i32, alloc, &one);
    defer alloc.free(r2);
    try testing.expectEqualSlices(i32, &[_]i32{9}, r2);
}

test "recursive quick sort: extreme random input" {
    const alloc = testing.allocator;
    const n: usize = 12_000;
    const arr = try alloc.alloc(i32, n);
    defer alloc.free(arr);

    var prng = std.Random.DefaultPrng.init(77);
    var random = prng.random();
    for (arr) |*v| {
        v.* = random.intRangeAtMost(i32, -1_000_000, 1_000_000);
    }

    const out = try recursiveQuickSort(i32, alloc, arr);
    defer alloc.free(out);
    try expectSortedAscending(i32, out);
}
