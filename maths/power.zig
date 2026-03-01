//! Binary Exponentiation (Fast Power) - Zig implementation
//! Reference: https://github.com/TheAlgorithms/Python/blob/master/maths/binary_exponentiation.py

const std = @import("std");
const testing = std.testing;

/// Computes base^exponent using binary exponentiation (iterative).
/// Time complexity: O(log n), Space complexity: O(1)
pub fn power(base: i64, exponent: u32) i64 {
    var result: i64 = 1;
    var b = base;
    var exp = exponent;
    while (exp > 0) {
        if (exp & 1 == 1) {
            result *= b;
        }
        b *= b;
        exp >>= 1;
    }
    return result;
}

/// Computes base^exponent mod modulus using modular exponentiation.
pub fn powerMod(base: u64, exponent: u64, modulus: u64) !u64 {
    if (modulus == 0) return error.InvalidModulus;
    if (modulus == 1) return 0;
    var result: u64 = 1;
    var b = base % modulus;
    var exp = exponent;
    while (exp > 0) {
        if (exp & 1 == 1) {
            result = mulMod(result, b, modulus);
        }
        b = mulMod(b, b, modulus);
        exp >>= 1;
    }
    return result;
}

fn mulMod(a: u64, b: u64, modulus: u64) u64 {
    return @intCast((@as(u128, a) * @as(u128, b)) % @as(u128, modulus));
}

test "power: basic cases" {
    try testing.expectEqual(@as(i64, 243), power(3, 5));
    try testing.expectEqual(@as(i64, -1), power(-1, 3));
    try testing.expectEqual(@as(i64, 0), power(0, 5));
    try testing.expectEqual(@as(i64, 3), power(3, 1));
    try testing.expectEqual(@as(i64, 1), power(3, 0));
    try testing.expectEqual(@as(i64, 1), power(1, 100));
}

test "power: larger exponent" {
    try testing.expectEqual(@as(i64, 1024), power(2, 10));
    try testing.expectEqual(@as(i64, 1048576), power(2, 20));
}

test "power mod: basic cases" {
    try testing.expectEqual(@as(u64, 1), try powerMod(3, 4, 5));
    try testing.expectEqual(@as(u64, 4), try powerMod(11, 13, 7));
    try testing.expectEqual(@as(u64, 0), try powerMod(5, 3, 1));
}

test "power mod: invalid modulus" {
    try testing.expectError(error.InvalidModulus, powerMod(7, 13, 0));
}

test "power mod: large multiplicands avoid intermediate overflow" {
    const base: u64 = std.math.maxInt(u64) - 3;
    const modulus: u64 = std.math.maxInt(u64) - 58;
    const reduced = base % modulus;
    const expected: u64 = @intCast((@as(u128, reduced) * @as(u128, reduced)) % @as(u128, modulus));
    try testing.expectEqual(expected, try powerMod(base, 2, modulus));
}
