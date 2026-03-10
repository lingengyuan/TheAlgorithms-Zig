//! Fibonacci Search - Zig implementation
//! Reference: https://github.com/TheAlgorithms/Python/blob/master/searches/fibonacci_search.py

const std = @import("std");
const testing = std.testing;

/// Returns the k-th Fibonacci number.
/// Time complexity: O(k), Space complexity: O(1)
pub fn fibonacci(k: usize) usize {
    if (k == 0) return 0;
    if (k == 1) return 1;

    var prev2: usize = 0;
    var prev1: usize = 1;
    var index: usize = 2;
    while (index <= k) : (index += 1) {
        const next = prev1 + prev2;
        prev2 = prev1;
        prev1 = next;
    }
    return prev1;
}

/// Returns the index of `target` in the sorted slice, or null when absent.
/// Time complexity: O(log n), Space complexity: O(1)
pub fn fibonacciSearch(comptime T: type, items: []const T, target: T) ?usize {
    if (items.len == 0) return null;

    var fib_mm2: usize = 0;
    var fib_mm1: usize = 1;
    var fib_m: usize = fib_mm1 + fib_mm2;
    while (fib_m < items.len) {
        fib_mm2 = fib_mm1;
        fib_mm1 = fib_m;
        fib_m = fib_mm1 + fib_mm2;
    }

    var offset: isize = -1;
    while (fib_m > 1) {
        const probe_offset = @as(usize, @intCast(offset + @as(isize, @intCast(fib_mm2))));
        const index = @min(probe_offset, items.len - 1);

        if (items[index] < target) {
            fib_m = fib_mm1;
            fib_mm1 = fib_mm2;
            fib_mm2 = fib_m - fib_mm1;
            offset = @as(isize, @intCast(index));
        } else if (target < items[index]) {
            fib_m = fib_mm2;
            fib_mm1 -= fib_mm2;
            fib_mm2 = fib_m - fib_mm1;
        } else {
            return index;
        }
    }

    const next_index = offset + 1;
    if (fib_mm1 == 1 and next_index >= 0) {
        const index = @as(usize, @intCast(next_index));
        if (index < items.len and items[index] == target) return index;
    }
    return null;
}

test "fibonacci search: fibonacci helper" {
    try testing.expectEqual(@as(usize, 0), fibonacci(0));
    try testing.expectEqual(@as(usize, 1), fibonacci(2));
    try testing.expectEqual(@as(usize, 5), fibonacci(5));
    try testing.expectEqual(@as(usize, 610), fibonacci(15));
}

test "fibonacci search: examples" {
    try testing.expectEqual(@as(?usize, 0), fibonacciSearch(i32, &[_]i32{ 4, 5, 6, 7 }, 4));
    try testing.expectEqual(@as(?usize, null), fibonacciSearch(i32, &[_]i32{ 4, 5, 6, 7 }, -10));
    try testing.expectEqual(@as(?usize, 0), fibonacciSearch(i32, &[_]i32{ -18, 2 }, -18));
    try testing.expectEqual(@as(?usize, 0), fibonacciSearch(i32, &[_]i32{5}, 5));
    try testing.expectEqual(@as(?usize, 1), fibonacciSearch(f64, &[_]f64{ 0.1, 0.4, 7.0 }, 0.4));
}

test "fibonacci search: boundaries and extremes" {
    try testing.expectEqual(@as(?usize, null), fibonacciSearch(i32, &[_]i32{}, 1));

    var ascending: [100]i32 = undefined;
    for (&ascending, 0..) |*item, index| item.* = @as(i32, @intCast(index));
    try testing.expectEqual(@as(?usize, 63), fibonacciSearch(i32, &ascending, 63));
    try testing.expectEqual(@as(?usize, 99), fibonacciSearch(i32, &ascending, 99));

    var stepped: [40]i32 = undefined;
    for (&stepped, 0..) |*item, index| item.* = -100 + @as(i32, @intCast(index * 5));
    try testing.expectEqual(@as(?usize, 20), fibonacciSearch(i32, &stepped, 0));
    try testing.expectEqual(@as(?usize, 39), fibonacciSearch(i32, &stepped, 95));
    try testing.expectEqual(@as(?usize, null), fibonacciSearch(i32, &stepped, 96));
}
