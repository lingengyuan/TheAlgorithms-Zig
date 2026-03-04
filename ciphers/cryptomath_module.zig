//! Cryptomath Module - Zig implementation
//! Reference: https://github.com/TheAlgorithms/Python/blob/master/ciphers/cryptomath_module.py

const std = @import("std");
const testing = std.testing;

pub const CryptoMathError = error{
    InvalidModulus,
    NoModInverse,
};

fn gcdByIterative(a_init: i128, b_init: i128) i128 {
    var a = if (a_init < 0) -a_init else a_init;
    var b = if (b_init < 0) -b_init else b_init;

    while (b != 0) {
        const t = @mod(a, b);
        a = b;
        b = t;
    }

    return a;
}

/// Returns modular inverse of `a (mod m)` via extended Euclid.
/// Time complexity: O(log m), Space complexity: O(1)
pub fn findModInverse(a: i128, m: i128) !i128 {
    if (m <= 0) return CryptoMathError.InvalidModulus;
    if (gcdByIterative(a, m) != 1) return CryptoMathError.NoModInverse;

    var s1: i128 = 1;
    var s2: i128 = 0;
    var s3: i128 = a;

    var t1: i128 = 0;
    var t2: i128 = 1;
    var t3: i128 = m;

    while (t3 != 0) {
        const q = @divTrunc(s3, t3);
        const next_t1 = s1 - q * t1;
        const next_t2 = s2 - q * t2;
        const next_t3 = s3 - q * t3;

        s1 = t1;
        s2 = t2;
        s3 = t3;

        t1 = next_t1;
        t2 = next_t2;
        t3 = next_t3;
    }

    return @mod(s1, m);
}

test "cryptomath: modular inverse basics" {
    try testing.expectEqual(@as(i128, 15), try findModInverse(7, 26));
    try testing.expectEqual(@as(i128, 2753), try findModInverse(17, 3120));
}

test "cryptomath: invalid and no inverse" {
    try testing.expectError(CryptoMathError.InvalidModulus, findModInverse(5, 0));
    try testing.expectError(CryptoMathError.NoModInverse, findModInverse(6, 9));
}

test "cryptomath: extreme large values" {
    const a: i128 = 9_223_372_036_854_775_561;
    const m: i128 = 9_223_372_036_854_775_583;

    const inv = try findModInverse(a, m);
    try testing.expectEqual(@as(i128, 1), @mod(a * inv, m));
}
