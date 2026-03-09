//! Modular Division - Zig implementation
//! Reference: https://github.com/TheAlgorithms/Python/blob/master/maths/modular_division.py

const std = @import("std");
const testing = std.testing;
const gcd_mod = @import("gcd.zig");
const ext_gcd_mod = @import("extended_euclidean.zig");
const modular_inverse_mod = @import("modular_inverse.zig");

pub const ModularDivisionError = error{
    InvalidInput,
    NoInverse,
};

/// Returns the greatest common divisor.
pub fn greatestCommonDivisor(a: i64, b: i64) u64 {
    return gcd_mod.gcd(a, b);
}

/// Returns `(gcd, x, y)` where `a*x + b*y = gcd(a, b)`.
pub fn extendedGcd(a: i64, b: i64) ext_gcd_mod.ExtendedGcdResult {
    return ext_gcd_mod.extendedEuclidean(a, b);
}

/// Returns `(x, y)` coefficients from the extended Euclidean algorithm.
pub fn extendedEuclid(a: i64, b: i64) struct { x: i64, y: i64 } {
    const result = ext_gcd_mod.extendedEuclidean(a, b);
    return .{ .x = result.x, .y = result.y };
}

/// Returns the modular inverse of `a mod n`.
pub fn invertModulo(a: i64, n: i64) !u64 {
    return modular_inverse_mod.modularInverse(a, n);
}

/// Solves `b / a (mod n)` by multiplying with the modular inverse of `a`.
pub fn modularDivision(a: i64, b: i64, n: i64) ModularDivisionError!u64 {
    if (n <= 1 or a <= 0) return error.InvalidInput;
    if (greatestCommonDivisor(a, n) != 1) return error.NoInverse;
    const inverse = invertModulo(a, n) catch return error.NoInverse;
    const b_mod: u64 = @intCast(@mod(b, n));
    return @intCast((@as(u128, b_mod) * @as(u128, inverse)) % @as(u128, @intCast(n)));
}

/// Alternative modular division implementation using `invertModulo`.
pub fn modularDivision2(a: i64, b: i64, n: i64) ModularDivisionError!u64 {
    return modularDivision(a, b, n);
}

test "modular division: python reference examples" {
    try testing.expectEqual(@as(u64, 2), try modularDivision(4, 8, 5));
    try testing.expectEqual(@as(u64, 1), try modularDivision(3, 8, 5));
    try testing.expectEqual(@as(u64, 4), try modularDivision(4, 11, 5));

    try testing.expectEqual(@as(u64, 2), try modularDivision2(4, 8, 5));
    try testing.expectEqual(@as(u64, 3), try invertModulo(2, 5));
}

test "modular division: edge and extreme cases" {
    const ext = extendedGcd(10, 6);
    try testing.expectEqual(@as(i64, 2), ext.gcd);
    const pair = extendedEuclid(7, 5);
    try testing.expectEqual(@as(i64, -2), pair.x);
    try testing.expectEqual(@as(i64, 3), pair.y);
    try testing.expectError(error.InvalidInput, modularDivision(0, 8, 5));
    try testing.expectError(error.NoInverse, modularDivision(6, 8, 12));
}
