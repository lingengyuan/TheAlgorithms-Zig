//! Sylvester Sequence - Zig implementation
//! Reference: https://github.com/TheAlgorithms/Python/blob/master/maths/sylvester_sequence.py

const std = @import("std");
const testing = std.testing;

pub const SylvesterError = error{ InvalidInput, Overflow };

/// Returns nth Sylvester number (1-indexed).
/// Time complexity: O(n) recursion, Space complexity: O(n)
pub fn sylvester(number: i64) SylvesterError!u128 {
    if (number < 1) return SylvesterError.InvalidInput;
    if (number == 1) return 2;

    const prev = try sylvester(number - 1);
    const lower = @subWithOverflow(prev, @as(u128, 1));
    if (lower[1] != 0) return SylvesterError.Overflow;
    const product = @mulWithOverflow(lower[0], prev);
    if (product[1] != 0) return SylvesterError.Overflow;
    const next = @addWithOverflow(product[0], @as(u128, 1));
    if (next[1] != 0) return SylvesterError.Overflow;
    return next[0];
}

test "sylvester sequence: python reference examples" {
    try testing.expectEqual(@as(u128, 113_423_713_055_421_844_361_000_443), try sylvester(8));
    try testing.expectError(SylvesterError.InvalidInput, sylvester(-1));
}

test "sylvester sequence: edge and extreme cases" {
    try testing.expectEqual(@as(u128, 2), try sylvester(1));
    try testing.expectEqual(@as(u128, 3), try sylvester(2));
    try testing.expectEqual(@as(u128, 7), try sylvester(3));
    try testing.expectError(SylvesterError.Overflow, sylvester(9));
}
