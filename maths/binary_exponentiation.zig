//! Binary Exponentiation - Zig implementation
//! Reference: https://github.com/TheAlgorithms/Python/blob/master/maths/binary_exponentiation.py

const std = @import("std");
const testing = std.testing;

pub const BinaryExponentiationError = error{
    InvalidExponent,
    InvalidModulus,
};

/// Computes `base^exponent` recursively.
/// Time complexity: O(log exponent), Space complexity: O(log exponent)
pub fn binaryExpRecursive(base: f64, exponent: i64) BinaryExponentiationError!f64 {
    if (exponent < 0) return error.InvalidExponent;
    if (exponent == 0) return 1;
    if (@rem(exponent, 2) == 1) {
        return try binaryExpRecursive(base, exponent - 1) * base;
    }
    const partial = try binaryExpRecursive(base, @divTrunc(exponent, 2));
    return partial * partial;
}

/// Computes `base^exponent` iteratively.
/// Time complexity: O(log exponent), Space complexity: O(1)
pub fn binaryExpIterative(base: f64, exponent: i64) BinaryExponentiationError!f64 {
    if (exponent < 0) return error.InvalidExponent;
    var result: f64 = 1;
    var current_base = base;
    var current_exp: u64 = @intCast(exponent);
    while (current_exp > 0) {
        if (current_exp & 1 == 1) result *= current_base;
        current_base *= current_base;
        current_exp >>= 1;
    }
    return result;
}

/// Computes `(base^exponent) mod modulus` recursively.
/// Time complexity: O(log exponent), Space complexity: O(log exponent)
pub fn binaryExpModRecursive(base: f64, exponent: i64, modulus: i64) BinaryExponentiationError!f64 {
    if (exponent < 0) return error.InvalidExponent;
    if (modulus <= 0) return error.InvalidModulus;
    if (exponent == 0) return 1;
    if (@rem(exponent, 2) == 1) {
        return @mod((try binaryExpModRecursive(base, exponent - 1, modulus)) * base, @as(f64, @floatFromInt(modulus)));
    }
    const partial = try binaryExpModRecursive(base, @divTrunc(exponent, 2), modulus);
    return @mod(partial * partial, @as(f64, @floatFromInt(modulus)));
}

/// Computes `(base^exponent) mod modulus` iteratively.
/// Time complexity: O(log exponent), Space complexity: O(1)
pub fn binaryExpModIterative(base: f64, exponent: i64, modulus: i64) BinaryExponentiationError!f64 {
    if (exponent < 0) return error.InvalidExponent;
    if (modulus <= 0) return error.InvalidModulus;

    const mod_f: f64 = @floatFromInt(modulus);
    var result: f64 = 1;
    var current_base = @mod(base, mod_f);
    var current_exp: u64 = @intCast(exponent);

    while (current_exp > 0) {
        if (current_exp & 1 == 1) {
            result = @mod(result * current_base, mod_f);
        }
        current_base = @mod(current_base * current_base, mod_f);
        current_exp >>= 1;
    }
    return result;
}

test "binary exponentiation: python reference examples" {
    try testing.expectApproxEqAbs(@as(f64, 243), try binaryExpRecursive(3, 5), 1e-12);
    try testing.expectApproxEqAbs(@as(f64, 243), try binaryExpIterative(3, 5), 1e-12);
    try testing.expectApproxEqAbs(@as(f64, 1), try binaryExpModRecursive(3, 4, 5), 1e-12);
    try testing.expectApproxEqAbs(@as(f64, 4), try binaryExpModIterative(11, 13, 7), 1e-12);
    try testing.expectApproxEqAbs(@as(f64, 2.0625), try binaryExpModIterative(1.5, 4, 3), 1e-12);
}

test "binary exponentiation: edge and extreme cases" {
    try testing.expectApproxEqAbs(@as(f64, 1), try binaryExpIterative(3, 0), 1e-12);
    try testing.expectApproxEqAbs(@as(f64, 5.0625), try binaryExpRecursive(1.5, 4), 1e-12);
    try testing.expectError(error.InvalidExponent, binaryExpRecursive(3, -1));
    try testing.expectError(error.InvalidModulus, binaryExpModRecursive(7, 13, 0));
}
