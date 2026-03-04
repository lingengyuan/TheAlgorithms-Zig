//! Binary Insertion Sort - Zig implementation
//! Reference: https://github.com/TheAlgorithms/Python/blob/master/sorts/binary_insertion_sort.py

const std = @import("std");
const testing = std.testing;

/// In-place binary insertion sort, ascending order.
/// Time complexity: O(n²), Space complexity: O(1)
pub fn binaryInsertionSort(comptime T: type, arr: []T) void {
    if (arr.len <= 1) return;

    var i: usize = 1;
    while (i < arr.len) : (i += 1) {
        const value = arr[i];
        var low: usize = 0;
        var high: usize = i;

        while (low < high) {
            const mid = low + (high - low) / 2;
            if (value < arr[mid]) {
                high = mid;
            } else {
                low = mid + 1;
            }
        }

        var j = i;
        while (j > low) : (j -= 1) {
            arr[j] = arr[j - 1];
        }
        arr[low] = value;
    }
}

fn expectSortedAscending(comptime T: type, arr: []const T) !void {
    if (arr.len <= 1) return;
    for (1..arr.len) |i| {
        try testing.expect(arr[i - 1] <= arr[i]);
    }
}

test "binary insertion sort: python reference examples" {
    var a1 = [_]i32{ 0, 4, 1234, 4, 1 };
    binaryInsertionSort(i32, &a1);
    try testing.expectEqualSlices(i32, &[_]i32{ 0, 1, 4, 4, 1234 }, &a1);

    var a2 = [_]i32{};
    binaryInsertionSort(i32, &a2);
    try testing.expectEqual(@as(usize, 0), a2.len);

    var a3 = [_]i32{ -1, -2, -3 };
    binaryInsertionSort(i32, &a3);
    try testing.expectEqualSlices(i32, &[_]i32{ -3, -2, -1 }, &a3);
}

test "binary insertion sort: edge cases" {
    var single = [_]i32{42};
    binaryInsertionSort(i32, &single);
    try testing.expectEqualSlices(i32, &[_]i32{42}, &single);

    var dup = [_]i32{ 5, 1, 5, 1, 5, 1 };
    binaryInsertionSort(i32, &dup);
    try testing.expectEqualSlices(i32, &[_]i32{ 1, 1, 1, 5, 5, 5 }, &dup);
}

test "binary insertion sort: extreme descending input" {
    const alloc = testing.allocator;
    const n: usize = 4000;
    const arr = try alloc.alloc(i32, n);
    defer alloc.free(arr);

    for (arr, 0..) |*v, idx| {
        v.* = @as(i32, @intCast(n - idx));
    }

    binaryInsertionSort(i32, arr);
    try expectSortedAscending(i32, arr);
    try testing.expectEqual(@as(i32, 1), arr[0]);
    try testing.expectEqual(@as(i32, @intCast(n)), arr[n - 1]);
}
