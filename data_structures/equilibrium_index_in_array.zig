//! Equilibrium Index In Array - Zig implementation
//! Reference: https://github.com/TheAlgorithms/Python/blob/master/data_structures/arrays/equilibrium_index_in_array.py

const std = @import("std");
const testing = std.testing;

/// Returns equilibrium index, or -1 if not found.
/// Time complexity: O(n), Space complexity: O(1)
pub fn equilibriumIndex(arr: []const i64) i64 {
    var total_sum: i128 = 0;
    for (arr) |v| total_sum += v;

    var left_sum: i128 = 0;
    for (arr, 0..) |value, i| {
        total_sum -= value;
        if (left_sum == total_sum) return @intCast(i);
        left_sum += value;
    }

    return -1;
}

test "equilibrium index in array: python examples" {
    try testing.expectEqual(@as(i64, 3), equilibriumIndex(&[_]i64{ -7, 1, 5, 2, -4, 3, 0 }));
    try testing.expectEqual(@as(i64, -1), equilibriumIndex(&[_]i64{ 1, 2, 3, 4, 5 }));
    try testing.expectEqual(@as(i64, 2), equilibriumIndex(&[_]i64{ 1, 1, 1, 1, 1 }));
    try testing.expectEqual(@as(i64, -1), equilibriumIndex(&[_]i64{ 2, 4, 6, 8, 10, 3 }));
}

test "equilibrium index in array: boundary and extreme" {
    try testing.expectEqual(@as(i64, -1), equilibriumIndex(&[_]i64{}));
    try testing.expectEqual(@as(i64, 0), equilibriumIndex(&[_]i64{0}));

    const n: usize = 100_001;
    const values = try testing.allocator.alloc(i64, n);
    defer testing.allocator.free(values);
    @memset(values, 1);

    try testing.expectEqual(@as(i64, 50_000), equilibriumIndex(values));
}
