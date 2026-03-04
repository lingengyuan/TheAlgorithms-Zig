//! Kth Largest Element - Zig implementation
//! Reference: https://github.com/TheAlgorithms/Python/blob/master/data_structures/arrays/kth_largest_element.py

const std = @import("std");
const testing = std.testing;

fn partition(arr: []i64, low: usize, high: usize) usize {
    const pivot = arr[high];
    var i = low;
    var j = low;
    while (j < high) : (j += 1) {
        if (arr[j] >= pivot) {
            std.mem.swap(i64, &arr[i], &arr[j]);
            i += 1;
        }
    }
    std.mem.swap(i64, &arr[i], &arr[high]);
    return i;
}

/// Finds kth largest element (1-based position). Empty input returns -1.
/// Time complexity: average O(n), Space complexity: O(n)
pub fn kthLargestElement(allocator: std.mem.Allocator, arr: []const i64, position: usize) !i64 {
    if (arr.len == 0) return -1;
    if (position < 1 or position > arr.len) return error.InvalidPosition;

    const copy = try allocator.alloc(i64, arr.len);
    defer allocator.free(copy);
    @memcpy(copy, arr);

    var low: usize = 0;
    var high: usize = copy.len - 1;
    const target = position - 1;

    while (low <= high) {
        const pivot_index = partition(copy, low, high);
        if (pivot_index == target) return copy[pivot_index];

        if (pivot_index > target) {
            if (pivot_index == 0) break;
            high = pivot_index - 1;
        } else {
            low = pivot_index + 1;
        }
    }

    return -1;
}

test "kth largest element: python examples" {
    try testing.expectEqual(@as(i64, 5), try kthLargestElement(testing.allocator, &[_]i64{ 3, 1, 4, 1, 5, 9, 2, 6, 5, 3, 5 }, 3));
    try testing.expectEqual(@as(i64, 9), try kthLargestElement(testing.allocator, &[_]i64{ 2, 5, 6, 1, 9, 3, 8, 4, 7, 3, 5 }, 1));
    try testing.expectEqual(@as(i64, -1), try kthLargestElement(testing.allocator, &[_]i64{}, 1));
    try testing.expectEqual(@as(i64, -1), try kthLargestElement(testing.allocator, &[_]i64{ -2, -5, -4, -1 }, 1));
}

test "kth largest element: invalid position" {
    try testing.expectError(error.InvalidPosition, kthLargestElement(testing.allocator, &[_]i64{ 2, 5, 6 }, 0));
    try testing.expectError(error.InvalidPosition, kthLargestElement(testing.allocator, &[_]i64{ 2, 5, 6 }, 4));
}

test "kth largest element: extreme" {
    const n: usize = 100_000;
    const values = try testing.allocator.alloc(i64, n);
    defer testing.allocator.free(values);

    for (0..n) |i| {
        values[i] = @intCast(i + 1);
    }

    var prng = std.Random.DefaultPrng.init(0x5EED_1234);
    const random = prng.random();
    var i = n;
    while (i > 1) : (i -= 1) {
        const j = random.uintLessThan(usize, i);
        std.mem.swap(i64, &values[i - 1], &values[j]);
    }

    try testing.expectEqual(@as(i64, @intCast(n)), try kthLargestElement(testing.allocator, values, 1));
    try testing.expectEqual(@as(i64, @intCast(n / 2)), try kthLargestElement(testing.allocator, values, n / 2 + 1));
    try testing.expectEqual(@as(i64, 1), try kthLargestElement(testing.allocator, values, n));
}
