//! Bitwise Addition (Recursive) - Zig implementation
//! Reference: https://github.com/TheAlgorithms/Python/blob/master/bit_manipulation/bitwise_addition_recursive.py

const std = @import("std");
const testing = std.testing;

pub const BitwiseAdditionError = error{ NegativeValue, Overflow };

fn addRecursiveU64(a: u64, b: u64) BitwiseAdditionError!u64 {
    if (b == 0) return a;

    const bitwise_sum = a ^ b;
    const carry = a & b;
    const shifted = @shlWithOverflow(carry, 1);
    if (shifted[1] != 0) return BitwiseAdditionError.Overflow;

    return addRecursiveU64(bitwise_sum, shifted[0]);
}

/// Adds two non-negative integers using only bitwise operations recursively.
///
/// API note: Python supports arbitrary-precision integers; this Zig version
/// returns `error.Overflow` when result exceeds `i64` range.
///
/// Time complexity: O(w), where w is the bit width.
/// Space complexity: O(w) recursion depth.
pub fn bitwiseAdditionRecursive(number: i64, other_number: i64) BitwiseAdditionError!i64 {
    if (number < 0 or other_number < 0) return BitwiseAdditionError.NegativeValue;

    const sum_u64 = try addRecursiveU64(@intCast(number), @intCast(other_number));
    if (sum_u64 > std.math.maxInt(i64)) return BitwiseAdditionError.Overflow;
    return @intCast(sum_u64);
}

test "bitwise addition recursive: python examples" {
    try testing.expectEqual(@as(i64, 9), try bitwiseAdditionRecursive(4, 5));
    try testing.expectEqual(@as(i64, 17), try bitwiseAdditionRecursive(8, 9));
    try testing.expectEqual(@as(i64, 4), try bitwiseAdditionRecursive(0, 4));
}

test "bitwise addition recursive: invalid negatives" {
    try testing.expectError(BitwiseAdditionError.NegativeValue, bitwiseAdditionRecursive(-1, 9));
    try testing.expectError(BitwiseAdditionError.NegativeValue, bitwiseAdditionRecursive(1, -9));
}

test "bitwise addition recursive: extreme bounds" {
    try testing.expectEqual(@as(i64, std.math.maxInt(i64)), try bitwiseAdditionRecursive(std.math.maxInt(i64) - 1, 1));
    try testing.expectError(BitwiseAdditionError.Overflow, bitwiseAdditionRecursive(std.math.maxInt(i64), 1));
}
