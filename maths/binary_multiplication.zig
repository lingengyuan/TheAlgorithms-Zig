//! Binary Multiplication - Zig implementation
//! Reference: https://github.com/TheAlgorithms/Python/blob/master/maths/binary_multiplication.py

const std = @import("std");
const testing = std.testing;

pub const BinaryMultiplicationError = error{InvalidModulus};

/// Multiplies two integers using doubling/halving.
/// Time complexity: O(log b), Space complexity: O(1)
pub fn binaryMultiply(a: i64, b: i64) i64 {
    if (b < 0) return -binaryMultiply(a, -b);
    var result: i64 = 0;
    var lhs = a;
    var rhs: u64 = @intCast(b);
    while (rhs > 0) {
        if (rhs & 1 == 1) result += lhs;
        lhs += lhs;
        rhs >>= 1;
    }
    return result;
}

/// Computes `(a * b) mod modulus`.
/// Time complexity: O(log b), Space complexity: O(1)
pub fn binaryModMultiply(a: i64, b: i64, modulus: i64) BinaryMultiplicationError!i64 {
    if (modulus <= 0) return error.InvalidModulus;

    var result: i64 = 0;
    var lhs = @mod(a, modulus);
    var rhs: u64 = if (b < 0) @intCast(-b) else @intCast(b);

    while (rhs > 0) {
        if (rhs & 1 == 1) {
            result = @mod(result + lhs, modulus);
        }
        lhs = @mod(lhs + lhs, modulus);
        rhs >>= 1;
    }
    return if (b < 0) @mod(-result, modulus) else result;
}

test "binary multiplication: python reference examples" {
    try testing.expectEqual(@as(i64, 6), binaryMultiply(2, 3));
    try testing.expectEqual(@as(i64, 12), binaryMultiply(3, 4));
    try testing.expectEqual(@as(i64, 50), binaryMultiply(10, 5));
    try testing.expectEqual(@as(i64, 1), try binaryModMultiply(2, 3, 5));
    try testing.expectEqual(@as(i64, 11), try binaryModMultiply(10, 5, 13));
}

test "binary multiplication: edge and extreme cases" {
    try testing.expectEqual(@as(i64, 0), binaryMultiply(5, 0));
    try testing.expectEqual(@as(i64, -12), binaryMultiply(-3, 4));
    try testing.expectEqual(@as(i64, 7), try binaryModMultiply(-3, 4, 19));
    try testing.expectError(error.InvalidModulus, binaryModMultiply(2, 3, 0));
}
