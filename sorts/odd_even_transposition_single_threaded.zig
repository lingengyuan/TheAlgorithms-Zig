//! Odd-Even Transposition Sort (Single-threaded) - Zig implementation
//! Reference: https://github.com/TheAlgorithms/Python/blob/master/sorts/odd_even_transposition_single_threaded.py

const std = @import("std");
const testing = std.testing;

/// In-place odd-even transposition sort.
/// Time complexity: O(n²), Space complexity: O(1)
pub fn oddEvenTranspositionSort(comptime T: type, arr: []T) void {
    const n = arr.len;
    for (0..n) |pass| {
        var i = pass % 2;
        while (i + 1 < n) : (i += 2) {
            if (arr[i + 1] < arr[i]) {
                std.mem.swap(T, &arr[i], &arr[i + 1]);
            }
        }
    }
}

fn expectSortedAscending(comptime T: type, arr: []const T) !void {
    if (arr.len <= 1) return;
    for (1..arr.len) |i| try testing.expect(arr[i - 1] <= arr[i]);
}

test "odd even transposition: python reference examples" {
    var a1 = [_]i32{ 5, 4, 3, 2, 1 };
    oddEvenTranspositionSort(i32, &a1);
    try testing.expectEqualSlices(i32, &[_]i32{ 1, 2, 3, 4, 5 }, &a1);

    var a2 = [_]i32{ 13, 11, 18, 0, -1 };
    oddEvenTranspositionSort(i32, &a2);
    try testing.expectEqualSlices(i32, &[_]i32{ -1, 0, 11, 13, 18 }, &a2);

    var a3 = [_]f64{ -0.1, 1.1, 0.1, -2.9 };
    oddEvenTranspositionSort(f64, &a3);
    try testing.expectEqualSlices(f64, &[_]f64{ -2.9, -0.1, 0.1, 1.1 }, &a3);
}

test "odd even transposition: edge cases" {
    var empty = [_]i32{};
    oddEvenTranspositionSort(i32, &empty);
    try testing.expectEqual(@as(usize, 0), empty.len);

    var one = [_]i32{42};
    oddEvenTranspositionSort(i32, &one);
    try testing.expectEqualSlices(i32, &[_]i32{42}, &one);
}

test "odd even transposition: extreme descending input" {
    const alloc = testing.allocator;
    const n: usize = 4000;
    const arr = try alloc.alloc(i32, n);
    defer alloc.free(arr);
    for (arr, 0..) |*v, i| v.* = @as(i32, @intCast(n - i));

    oddEvenTranspositionSort(i32, arr);
    try expectSortedAscending(i32, arr);
}
