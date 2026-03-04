//! Iterative Merge Sort - Zig implementation
//! Reference: https://github.com/TheAlgorithms/Python/blob/master/sorts/iterative_merge_sort.py

const std = @import("std");
const testing = std.testing;

fn min(a: usize, b: usize) usize {
    return if (a < b) a else b;
}

/// In-place iterative merge sort using O(n) temporary buffer.
/// Time complexity: O(n log n), Space complexity: O(n)
pub fn iterativeMergeSort(comptime T: type, allocator: std.mem.Allocator, arr: []T) !void {
    if (arr.len <= 1) return;

    const temp = try allocator.alloc(T, arr.len);
    defer allocator.free(temp);

    var width: usize = 1;
    while (width < arr.len) : (width *= 2) {
        var left: usize = 0;
        while (left < arr.len) : (left += 2 * width) {
            const mid = min(left + width, arr.len);
            const right = min(left + 2 * width, arr.len);
            if (mid == right) continue;

            var i = left;
            var j = mid;
            var k = left;

            while (i < mid and j < right) {
                if (arr[i] <= arr[j]) {
                    temp[k] = arr[i];
                    i += 1;
                } else {
                    temp[k] = arr[j];
                    j += 1;
                }
                k += 1;
            }

            while (i < mid) : (i += 1) {
                temp[k] = arr[i];
                k += 1;
            }
            while (j < right) : (j += 1) {
                temp[k] = arr[j];
                k += 1;
            }

            @memcpy(arr[left..right], temp[left..right]);
        }
    }
}

fn expectSortedAscending(comptime T: type, arr: []const T) !void {
    if (arr.len <= 1) return;
    for (1..arr.len) |i| {
        try testing.expect(arr[i - 1] <= arr[i]);
    }
}

test "iterative merge sort: python reference examples" {
    const alloc = testing.allocator;

    var a1 = [_]i32{ 5, 9, 8, 7, 1, 2, 7 };
    try iterativeMergeSort(i32, alloc, &a1);
    try testing.expectEqualSlices(i32, &[_]i32{ 1, 2, 5, 7, 7, 8, 9 }, &a1);

    var a2 = [_]i32{1};
    try iterativeMergeSort(i32, alloc, &a2);
    try testing.expectEqualSlices(i32, &[_]i32{1}, &a2);

    var a3 = [_]u8{ 'c', 'b', 'a' };
    try iterativeMergeSort(u8, alloc, &a3);
    try testing.expectEqualSlices(u8, &[_]u8{ 'a', 'b', 'c' }, &a3);

    var a4 = [_]f64{ 0.3, 0.2, 0.1 };
    try iterativeMergeSort(f64, alloc, &a4);
    try testing.expectEqualSlices(f64, &[_]f64{ 0.1, 0.2, 0.3 }, &a4);
}

test "iterative merge sort: edge cases" {
    const alloc = testing.allocator;

    var empty = [_]i32{};
    try iterativeMergeSort(i32, alloc, &empty);
    try testing.expectEqual(@as(usize, 0), empty.len);

    var mix = [_]f64{ 1.1, 1.0, 0.0, -1.0, -1.1 };
    try iterativeMergeSort(f64, alloc, &mix);
    try testing.expectEqualSlices(f64, &[_]f64{ -1.1, -1.0, 0.0, 1.0, 1.1 }, &mix);
}

test "iterative merge sort: extreme random input" {
    const alloc = testing.allocator;
    const n: usize = 20_000;
    const arr = try alloc.alloc(i32, n);
    defer alloc.free(arr);

    var prng = std.Random.DefaultPrng.init(99);
    var random = prng.random();
    for (arr) |*v| {
        v.* = random.intRangeAtMost(i32, -100_000, 100_000);
    }

    try iterativeMergeSort(i32, alloc, arr);
    try expectSortedAscending(i32, arr);
}
