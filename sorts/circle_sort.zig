//! Circle Sort - Zig implementation
//! Reference: https://github.com/TheAlgorithms/Python/blob/master/sorts/circle_sort.py

const std = @import("std");
const testing = std.testing;

fn circleSortUtil(comptime T: type, arr: []T, low: usize, high: usize) bool {
    var swapped = false;
    if (low == high) return swapped;

    var left = low;
    var right = high;

    while (left < right) {
        if (arr[left] > arr[right]) {
            std.mem.swap(T, &arr[left], &arr[right]);
            swapped = true;
        }
        left += 1;
        right -= 1;
    }

    if (left == right and right + 1 <= high and arr[left] > arr[right + 1]) {
        std.mem.swap(T, &arr[left], &arr[right + 1]);
        swapped = true;
    }

    const mid = low + (high - low) / 2;
    const left_swapped = circleSortUtil(T, arr, low, mid);
    const right_swapped = circleSortUtil(T, arr, mid + 1, high);

    return swapped or left_swapped or right_swapped;
}

/// In-place circle sort, ascending order.
/// Time complexity: O(n log n) average, O(n²) worst
/// Space complexity: O(log n) recursion
pub fn circleSort(comptime T: type, arr: []T) void {
    if (arr.len < 2) return;

    var is_not_sorted = true;
    while (is_not_sorted) {
        is_not_sorted = circleSortUtil(T, arr, 0, arr.len - 1);
    }
}

fn expectSortedAscending(comptime T: type, arr: []const T) !void {
    if (arr.len <= 1) return;
    for (1..arr.len) |i| try testing.expect(arr[i - 1] <= arr[i]);
}

test "circle sort: python reference examples" {
    var a1 = [_]i32{ 0, 5, 3, 2, 2 };
    circleSort(i32, &a1);
    try testing.expectEqualSlices(i32, &[_]i32{ 0, 2, 2, 3, 5 }, &a1);

    var a2 = [_]i32{};
    circleSort(i32, &a2);
    try testing.expectEqual(@as(usize, 0), a2.len);

    var a3 = [_]i32{ -2, 5, 0, -45 };
    circleSort(i32, &a3);
    try testing.expectEqualSlices(i32, &[_]i32{ -45, -2, 0, 5 }, &a3);
}

test "circle sort: utility behavior and edge cases" {
    var arr = [_]i32{ 5, 4, 3, 2, 1 };
    const swapped = circleSortUtil(i32, &arr, 0, 2);
    try testing.expect(swapped);
    try testing.expectEqualSlices(i32, &[_]i32{ 3, 4, 5, 2, 1 }, &arr);

    var one = [_]i32{9};
    circleSort(i32, &one);
    try testing.expectEqualSlices(i32, &[_]i32{9}, &one);
}

test "circle sort: extreme random input" {
    const alloc = testing.allocator;
    const n: usize = 15_000;
    const arr = try alloc.alloc(i32, n);
    defer alloc.free(arr);

    var prng = std.Random.DefaultPrng.init(300);
    var random = prng.random();
    for (arr) |*v| v.* = random.intRangeAtMost(i32, -100_000, 100_000);

    circleSort(i32, arr);
    try expectSortedAscending(i32, arr);
}
