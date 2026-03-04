//! Hexagonal Number - Zig implementation
//! Reference: https://github.com/TheAlgorithms/Python/blob/master/maths/special_numbers/hexagonal_number.py

const std = @import("std");
const testing = std.testing;

pub const HexagonalError = error{ InvalidInput, Overflow };

/// Returns the nth hexagonal number (`n * (2n - 1)`), for `n >= 1`.
/// Time complexity: O(1), Space complexity: O(1)
pub fn hexagonal(number: i64) HexagonalError!u128 {
    if (number < 1) return HexagonalError.InvalidInput;

    const n: u128 = @intCast(number);
    const two_n = @mulWithOverflow(@as(u128, 2), n);
    if (two_n[1] != 0) return HexagonalError.Overflow;
    const factor = @subWithOverflow(two_n[0], @as(u128, 1));
    if (factor[1] != 0) return HexagonalError.Overflow;
    const out = @mulWithOverflow(n, factor[0]);
    if (out[1] != 0) return HexagonalError.Overflow;
    return out[0];
}

test "hexagonal number: python reference examples" {
    try testing.expectEqual(@as(u128, 28), try hexagonal(4));
    try testing.expectEqual(@as(u128, 231), try hexagonal(11));
    try testing.expectEqual(@as(u128, 946), try hexagonal(22));
}

test "hexagonal number: invalid and extreme cases" {
    try testing.expectError(HexagonalError.InvalidInput, hexagonal(0));
    try testing.expectError(HexagonalError.InvalidInput, hexagonal(-1));

    const extreme = try hexagonal(std.math.maxInt(i64));
    try testing.expect(extreme > 0);
}
