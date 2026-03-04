//! Quick Sort 3 Partition - Zig implementation
//! Reference: https://github.com/TheAlgorithms/Python/blob/master/sorts/quick_sort_3_partition.py

const std = @import("std");
const testing = std.testing;

fn quickSort3PartitionRange(comptime T: type, arr: []T, left: usize, right: usize) void {
    if (right <= left) return;

    var a = left;
    var i = left;
    var b = right;
    const pivot = arr[left];

    while (i <= b) {
        if (arr[i] < pivot) {
            std.mem.swap(T, &arr[a], &arr[i]);
            a += 1;
            i += 1;
        } else if (arr[i] > pivot) {
            std.mem.swap(T, &arr[b], &arr[i]);
            if (b == 0) break;
            b -= 1;
        } else {
            i += 1;
        }
    }

    if (a > left) quickSort3PartitionRange(T, arr, left, a - 1);
    if (b < right) quickSort3PartitionRange(T, arr, b + 1, right);
}

/// In-place quick sort with Dutch-national-flag 3-way partitioning.
/// Time complexity: O(n log n) average, O(n²) worst
/// Space complexity: O(log n) recursion average
pub fn quickSort3Partition(comptime T: type, arr: []T) void {
    if (arr.len == 0) return;
    quickSort3PartitionRange(T, arr, 0, arr.len - 1);
}

test "quick sort 3 partition: python reference examples" {
    var array1 = [_]i32{ 5, -1, -1, 5, 5, 24, 0 };
    quickSort3Partition(i32, &array1);
    try testing.expectEqualSlices(i32, &[_]i32{ -1, -1, 0, 5, 5, 5, 24 }, &array1);

    var array2 = [_]i32{ 9, 0, 2, 6 };
    quickSort3Partition(i32, &array2);
    try testing.expectEqualSlices(i32, &[_]i32{ 0, 2, 6, 9 }, &array2);

    var array3 = [_]i32{};
    quickSort3Partition(i32, &array3);
    try testing.expectEqual(@as(usize, 0), array3.len);
}

test "quick sort 3 partition: edge cases" {
    var one = [_]i32{1};
    quickSort3Partition(i32, &one);
    try testing.expectEqualSlices(i32, &[_]i32{1}, &one);

    var dup = [_]i32{ 3, 3, 3, 3, 3 };
    quickSort3Partition(i32, &dup);
    try testing.expectEqualSlices(i32, &[_]i32{ 3, 3, 3, 3, 3 }, &dup);
}

test "quick sort 3 partition: extreme random input" {
    const alloc = testing.allocator;
    const n: usize = 25_000;
    const arr = try alloc.alloc(i32, n);
    defer alloc.free(arr);

    var prng = std.Random.DefaultPrng.init(2026);
    var random = prng.random();
    for (arr) |*v| {
        v.* = random.intRangeAtMost(i32, -5000, 5000);
    }

    quickSort3Partition(i32, arr);
    for (1..arr.len) |i| {
        try testing.expect(arr[i - 1] <= arr[i]);
    }
}
