//! Modular Inverse - Zig implementation
//! Reference: https://github.com/TheAlgorithms/Python/blob/master/maths/modular_division.py

const std = @import("std");
const testing = std.testing;
const ext_gcd = @import("extended_euclidean.zig");

/// Returns modular inverse of `a` modulo `m`, if it exists.
/// Requires `m > 1` and gcd(a, m) == 1.
/// Time complexity: O(log m), Space complexity: O(1)
pub fn modularInverse(a: i64, m: i64) !u64 {
    if (m <= 1) return error.InvalidModulus;

    const result = ext_gcd.extendedEuclidean(a, m);
    if (result.gcd != 1) return error.NoInverse;

    const m_i128: i128 = m;
    var inv: i128 = result.x;
    inv = @mod(inv, m_i128);
    if (inv < 0) inv += m_i128;
    return @intCast(inv);
}

test "modular inverse: basic case" {
    try testing.expectEqual(@as(u64, 4), try modularInverse(3, 11)); // 3*4 % 11 == 1
}

test "modular inverse: larger values" {
    try testing.expectEqual(@as(u64, 2753), try modularInverse(17, 3120));
}

test "modular inverse: negative a" {
    try testing.expectEqual(@as(u64, 7), try modularInverse(-3, 11)); // (-3)*7 % 11 == 1
}

test "modular inverse: no inverse" {
    try testing.expectError(error.NoInverse, modularInverse(6, 12));
}

test "modular inverse: invalid modulus" {
    try testing.expectError(error.InvalidModulus, modularInverse(3, 1));
    try testing.expectError(error.InvalidModulus, modularInverse(3, 0));
}
