//! Count Number of One Bits - Zig implementation
//! Reference: https://github.com/TheAlgorithms/Python/blob/master/bit_manipulation/count_number_of_one_bits.py

const std = @import("std");
const testing = std.testing;

pub const CountOneBitsError = error{NegativeValue};

/// Counts set bits using Brian Kernighan's algorithm.
///
/// Time complexity: O(k), k = number of set bits
/// Space complexity: O(1)
pub fn getSetBitsCountUsingBrianKernighansAlgorithm(number: i64) CountOneBitsError!u32 {
    if (number < 0) return CountOneBitsError.NegativeValue;

    var n: u64 = @intCast(number);
    var result: u32 = 0;
    while (n != 0) {
        n &= n - 1;
        result += 1;
    }
    return result;
}

/// Counts set bits by repeated modulo/division.
///
/// Time complexity: O(w), w = bit width
/// Space complexity: O(1)
pub fn getSetBitsCountUsingModuloOperator(number: i64) CountOneBitsError!u32 {
    if (number < 0) return CountOneBitsError.NegativeValue;

    var n: u64 = @intCast(number);
    var result: u32 = 0;
    while (n != 0) {
        if (n % 2 == 1) result += 1;
        n >>= 1;
    }
    return result;
}

test "count number of one bits: python examples for both methods" {
    const values = [_]i64{ 25, 37, 21, 58, 0, 256 };
    const expected = [_]u32{ 3, 3, 3, 4, 0, 1 };

    for (values, expected) |value, want| {
        try testing.expectEqual(want, try getSetBitsCountUsingBrianKernighansAlgorithm(value));
        try testing.expectEqual(want, try getSetBitsCountUsingModuloOperator(value));
    }
}

test "count number of one bits: invalid and equivalence extreme" {
    try testing.expectError(CountOneBitsError.NegativeValue, getSetBitsCountUsingBrianKernighansAlgorithm(-1));
    try testing.expectError(CountOneBitsError.NegativeValue, getSetBitsCountUsingModuloOperator(-1));

    const max_value = std.math.maxInt(i64);
    const c1 = try getSetBitsCountUsingBrianKernighansAlgorithm(max_value);
    const c2 = try getSetBitsCountUsingModuloOperator(max_value);
    try testing.expectEqual(c1, c2);
    try testing.expectEqual(@as(u32, 63), c1);
}
