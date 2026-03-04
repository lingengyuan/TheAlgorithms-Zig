//! Shrink Shell Sort - Zig implementation
//! Reference: https://github.com/TheAlgorithms/Python/blob/master/sorts/shrink_shell_sort.py

const std = @import("std");
const testing = std.testing;

/// In-place shell sort using gap shrink factor 1.3.
/// Time complexity: depends on gap sequence, typically sub-quadratic
/// Space complexity: O(1)
pub fn shrinkShellSort(comptime T: type, arr: []T) void {
    if (arr.len <= 1) return;

    var gap = arr.len;
    while (gap > 1) {
        const next = @as(usize, @intFromFloat(@floor(@as(f64, @floatFromInt(gap)) / 1.3)));
        gap = if (next < 1) 1 else next;

        var i: usize = gap;
        while (i < arr.len) : (i += 1) {
            const temp = arr[i];
            var j = i;
            while (j >= gap and arr[j - gap] > temp) {
                arr[j] = arr[j - gap];
                j -= gap;
            }
            arr[j] = temp;
        }
    }
}

fn expectSortedAscending(comptime T: type, arr: []const T) !void {
    if (arr.len <= 1) return;
    for (1..arr.len) |i| {
        try testing.expect(arr[i - 1] <= arr[i]);
    }
}

test "shrink shell sort: python reference examples" {
    var a1 = [_]i32{ 3, 2, 1 };
    shrinkShellSort(i32, &a1);
    try testing.expectEqualSlices(i32, &[_]i32{ 1, 2, 3 }, &a1);

    var a2 = [_]i32{};
    shrinkShellSort(i32, &a2);
    try testing.expectEqual(@as(usize, 0), a2.len);

    var a3 = [_]i32{1};
    shrinkShellSort(i32, &a3);
    try testing.expectEqualSlices(i32, &[_]i32{1}, &a3);
}

test "shrink shell sort: edge cases" {
    var dup = [_]i32{ 5, 5, 5, 5 };
    shrinkShellSort(i32, &dup);
    try testing.expectEqualSlices(i32, &[_]i32{ 5, 5, 5, 5 }, &dup);

    var mixed = [_]i32{ 9, -1, 7, 0, -3, 2 };
    shrinkShellSort(i32, &mixed);
    try testing.expectEqualSlices(i32, &[_]i32{ -3, -1, 0, 2, 7, 9 }, &mixed);
}

test "shrink shell sort: extreme descending input" {
    const alloc = testing.allocator;
    const n: usize = 20_000;
    const arr = try alloc.alloc(i32, n);
    defer alloc.free(arr);

    for (arr, 0..) |*v, i| {
        v.* = @as(i32, @intCast(n - i));
    }

    shrinkShellSort(i32, arr);
    try expectSortedAscending(i32, arr);
}
