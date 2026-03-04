//! Polygonal Numbers - Zig implementation
//! Reference: https://github.com/TheAlgorithms/Python/blob/master/maths/special_numbers/polygonal_numbers.py

const std = @import("std");
const testing = std.testing;

pub const PolygonalError = error{ InvalidInput, Overflow };

/// Returns `num`th `sides`-gonal number for `num >= 0` and `sides >= 3`.
/// Time complexity: O(1), Space complexity: O(1)
pub fn polygonalNum(num: i64, sides: i64) PolygonalError!u128 {
    if (num < 0 or sides < 3) return PolygonalError.InvalidInput;

    const n: u128 = @intCast(num);
    const s: u128 = @intCast(sides);

    if (sides == 3) {
        const add = @addWithOverflow(n, @as(u128, 1));
        if (add[1] != 0) return PolygonalError.Overflow;
        const mul = @mulWithOverflow(n, add[0]);
        if (mul[1] != 0) return PolygonalError.Overflow;
        return mul[0] / 2;
    }

    const n_square = @mulWithOverflow(n, n);
    if (n_square[1] != 0) return PolygonalError.Overflow;

    const s_minus_2 = @subWithOverflow(s, @as(u128, 2));
    if (s_minus_2[1] != 0) return PolygonalError.Overflow;
    const a = @mulWithOverflow(s_minus_2[0], n_square[0]);
    if (a[1] != 0) return PolygonalError.Overflow;

    const s_minus_4 = @subWithOverflow(s, @as(u128, 4));
    if (s_minus_4[1] != 0) return PolygonalError.Overflow;
    const b = @mulWithOverflow(s_minus_4[0], n);
    if (b[1] != 0) return PolygonalError.Overflow;

    const numerator = @subWithOverflow(a[0], b[0]);
    if (numerator[1] != 0) return PolygonalError.Overflow;
    return numerator[0] / 2;
}

test "polygonal numbers: python reference examples" {
    try testing.expectEqual(@as(u128, 0), try polygonalNum(0, 3));
    try testing.expectEqual(@as(u128, 6), try polygonalNum(3, 3));
    try testing.expectEqual(@as(u128, 25), try polygonalNum(5, 4));
    try testing.expectEqual(@as(u128, 5), try polygonalNum(2, 5));
}

test "polygonal numbers: invalid and extreme cases" {
    try testing.expectError(PolygonalError.InvalidInput, polygonalNum(-1, 0));
    try testing.expectError(PolygonalError.InvalidInput, polygonalNum(0, 2));

    const triangular_extreme = try polygonalNum(std.math.maxInt(i64), 3);
    try testing.expect(triangular_extreme > 0);

    try testing.expectError(PolygonalError.Overflow, polygonalNum(std.math.maxInt(i64), std.math.maxInt(i64)));
}
