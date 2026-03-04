//! Cycle Sort - Zig implementation
//! Reference: https://github.com/TheAlgorithms/Python/blob/master/sorts/cycle_sort.py

const std = @import("std");
const testing = std.testing;

/// In-place cycle sort, ascending order.
/// Time complexity: O(n²), Space complexity: O(1)
pub fn cycleSort(comptime T: type, arr: []T) void {
    if (arr.len <= 1) return;

    var cycle_start: usize = 0;
    while (cycle_start + 1 < arr.len) : (cycle_start += 1) {
        var item = arr[cycle_start];
        var pos = cycle_start;

        var i = cycle_start + 1;
        while (i < arr.len) : (i += 1) {
            if (arr[i] < item) pos += 1;
        }

        if (pos == cycle_start) continue;

        while (item == arr[pos]) {
            pos += 1;
        }

        {
            const tmp = arr[pos];
            arr[pos] = item;
            item = tmp;
        }

        while (pos != cycle_start) {
            pos = cycle_start;
            i = cycle_start + 1;
            while (i < arr.len) : (i += 1) {
                if (arr[i] < item) pos += 1;
            }

            while (item == arr[pos]) {
                pos += 1;
            }

            const tmp = arr[pos];
            arr[pos] = item;
            item = tmp;
        }
    }
}

fn expectSortedAscending(comptime T: type, arr: []const T) !void {
    if (arr.len <= 1) return;
    for (1..arr.len) |i| {
        try testing.expect(arr[i - 1] <= arr[i]);
    }
}

test "cycle sort: python reference examples" {
    var a1 = [_]i32{ 4, 3, 2, 1 };
    cycleSort(i32, &a1);
    try testing.expectEqualSlices(i32, &[_]i32{ 1, 2, 3, 4 }, &a1);

    var a2 = [_]i32{ -4, 20, 0, -50, 100, -1 };
    cycleSort(i32, &a2);
    try testing.expectEqualSlices(i32, &[_]i32{ -50, -4, -1, 0, 20, 100 }, &a2);

    var a3 = [_]f64{ -0.1, -0.2, 1.3, -0.8 };
    cycleSort(f64, &a3);
    try testing.expectEqualSlices(f64, &[_]f64{ -0.8, -0.2, -0.1, 1.3 }, &a3);
}

test "cycle sort: edge and duplicate cases" {
    var a1 = [_]i32{};
    cycleSort(i32, &a1);
    try testing.expectEqual(@as(usize, 0), a1.len);

    var a2 = [_]i32{ 3, 1, 3, 1, 2 };
    cycleSort(i32, &a2);
    try testing.expectEqualSlices(i32, &[_]i32{ 1, 1, 2, 3, 3 }, &a2);
}

test "cycle sort: extreme mixed input" {
    const alloc = testing.allocator;
    const n: usize = 3000;
    const arr = try alloc.alloc(i32, n);
    defer alloc.free(arr);

    for (arr, 0..) |*v, i| {
        const parity: i32 = @intCast(i % 2);
        if (parity == 0) {
            v.* = @as(i32, @intCast(n - i));
        } else {
            v.* = @as(i32, @intCast(i));
        }
    }

    cycleSort(i32, arr);
    try expectSortedAscending(i32, arr);
}
