//! Odd-Even Transposition Sort (Parallel model) - Zig implementation
//! Reference: https://github.com/TheAlgorithms/Python/blob/master/sorts/odd_even_transposition_parallel.py

const std = @import("std");
const testing = std.testing;

/// Sequential simulation of odd-even transposition passes.
/// Matches deterministic sorted outcome of Python implementation.
/// Time complexity: O(n²), Space complexity: O(1)
fn lessThan(comptime T: type, lhs: T, rhs: T) bool {
    if (T == bool) return @intFromBool(lhs) < @intFromBool(rhs);
    return lhs < rhs;
}

pub fn oddEvenTranspositionParallel(comptime T: type, arr: []T) void {
    const arr_size = arr.len;
    for (0..arr_size) |pass| {
        var i = pass % 2;
        while (i + 1 < arr_size) : (i += 2) {
            if (lessThan(T, arr[i + 1], arr[i])) {
                std.mem.swap(T, &arr[i], &arr[i + 1]);
            }
        }
    }
}

fn expectSortedAscending(comptime T: type, arr: []const T) !void {
    if (arr.len <= 1) return;
    for (1..arr.len) |i| try testing.expect(arr[i - 1] <= arr[i]);
}

test "odd even transposition parallel: python reference examples" {
    var a1 = [_]i32{ 10, 9, 8, 7, 6, 5, 4, 3, 2, 1 };
    oddEvenTranspositionParallel(i32, &a1);
    try testing.expectEqualSlices(i32, &[_]i32{ 1, 2, 3, 4, 5, 6, 7, 8, 9, 10 }, &a1);

    var a2 = [_]u8{ 'a', 'x', 'c' };
    oddEvenTranspositionParallel(u8, &a2);
    try testing.expectEqualSlices(u8, &[_]u8{ 'a', 'c', 'x' }, &a2);

    var a3 = [_]f64{ 1.9, 42.0, 2.8 };
    oddEvenTranspositionParallel(f64, &a3);
    try testing.expectEqualSlices(f64, &[_]f64{ 1.9, 2.8, 42.0 }, &a3);

    var a4 = [_]bool{ false, true, false };
    oddEvenTranspositionParallel(bool, &a4);
    try testing.expectEqualSlices(bool, &[_]bool{ false, false, true }, &a4);
}

test "odd even transposition parallel: edge cases" {
    var one = [_]i32{1};
    oddEvenTranspositionParallel(i32, &one);
    try testing.expectEqualSlices(i32, &[_]i32{1}, &one);

    var empty = [_]i32{};
    oddEvenTranspositionParallel(i32, &empty);
    try testing.expectEqual(@as(usize, 0), empty.len);
}

test "odd even transposition parallel: extreme random input" {
    const alloc = testing.allocator;
    const n: usize = 3500;
    const arr = try alloc.alloc(i32, n);
    defer alloc.free(arr);

    var prng = std.Random.DefaultPrng.init(555);
    var random = prng.random();
    for (arr) |*v| v.* = random.intRangeAtMost(i32, -1_000_000, 1_000_000);

    oddEvenTranspositionParallel(i32, arr);
    try expectSortedAscending(i32, arr);
}
