//! Triangular Numbers - Zig implementation
//! Reference: https://github.com/TheAlgorithms/Python/blob/master/maths/special_numbers/triangular_numbers.py

const std = @import("std");
const testing = std.testing;

pub const TriangularError = error{ InvalidInput, Overflow };

/// Returns triangular number at `position` (`position * (position + 1) / 2`).
/// Time complexity: O(1), Space complexity: O(1)
pub fn triangularNumber(position: i64) TriangularError!u128 {
    if (position < 0) return TriangularError.InvalidInput;

    const n: u128 = @intCast(position);
    const add = @addWithOverflow(n, @as(u128, 1));
    if (add[1] != 0) return TriangularError.Overflow;
    const mul = @mulWithOverflow(n, add[0]);
    if (mul[1] != 0) return TriangularError.Overflow;
    return mul[0] / 2;
}

test "triangular numbers: python reference examples" {
    try testing.expectEqual(@as(u128, 1), try triangularNumber(1));
    try testing.expectEqual(@as(u128, 6), try triangularNumber(3));
    try testing.expectError(TriangularError.InvalidInput, triangularNumber(-1));
}

test "triangular numbers: boundary and extreme cases" {
    try testing.expectEqual(@as(u128, 0), try triangularNumber(0));

    const extreme = try triangularNumber(std.math.maxInt(i64));
    try testing.expect(extreme > 0);
}
