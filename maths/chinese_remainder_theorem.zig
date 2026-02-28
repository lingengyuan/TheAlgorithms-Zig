//! Chinese Remainder Theorem - Zig implementation
//! Reference: https://github.com/TheAlgorithms/Python/blob/master/maths/chinese_remainder_theorem.py

const std = @import("std");
const testing = std.testing;
const mod_inv = @import("modular_inverse.zig");

/// Solves x ≡ remainders[i] (mod moduli[i]) for pairwise-coprime positive moduli.
/// Returns the smallest non-negative solution.
/// Time complexity: O(k^2 + k log M), Space complexity: O(1)
pub fn chineseRemainderTheorem(remainders: []const i64, moduli: []const i64) !i64 {
    if (remainders.len == 0 or moduli.len == 0) return error.EmptyInput;
    if (remainders.len != moduli.len) return error.LengthMismatch;

    const n = remainders.len;
    for (moduli) |m| {
        if (m <= 0) return error.InvalidModulus;
    }
    for (0..n) |i| {
        for (i + 1..n) |j| {
            if (gcdU64(@intCast(moduli[i]), @intCast(moduli[j])) != 1) {
                return error.NotPairwiseCoprime;
            }
        }
    }

    var prod: i128 = 1;
    for (moduli) |m| {
        const mul = @mulWithOverflow(prod, @as(i128, m));
        if (mul[1] != 0) return error.Overflow;
        prod = mul[0];
    }

    var sum: i128 = 0;
    for (remainders, moduli) |a_raw, m| {
        const a = @mod(@as(i128, a_raw), @as(i128, m));
        const ni: i128 = @divTrunc(prod, @as(i128, m));
        const ni_mod_m: i64 = @intCast(@mod(ni, @as(i128, m)));
        const inv = try mod_inv.modularInverse(ni_mod_m, m);

        const t1 = @mulWithOverflow(a, ni);
        if (t1[1] != 0) return error.Overflow;
        const t2 = @mulWithOverflow(t1[0], @as(i128, inv));
        if (t2[1] != 0) return error.Overflow;
        const add = @addWithOverflow(sum, t2[0]);
        if (add[1] != 0) return error.Overflow;
        sum = add[0];
    }

    const result = @mod(sum, prod);
    if (result > std.math.maxInt(i64)) return error.Overflow;
    return @intCast(result);
}

fn gcdU64(a: u64, b: u64) u64 {
    var x = a;
    var y = b;
    while (y != 0) {
        const t = y;
        y = x % y;
        x = t;
    }
    return x;
}

test "crt: classic example" {
    const rem = [_]i64{ 2, 3, 2 };
    const mod = [_]i64{ 3, 5, 7 };
    try testing.expectEqual(@as(i64, 23), try chineseRemainderTheorem(&rem, &mod));
}

test "crt: two equations" {
    const rem = [_]i64{ 1, 3 };
    const mod = [_]i64{ 2, 5 };
    try testing.expectEqual(@as(i64, 3), try chineseRemainderTheorem(&rem, &mod));
}

test "crt: non-coprime moduli returns error" {
    const rem = [_]i64{ 1, 2 };
    const mod = [_]i64{ 4, 6 };
    try testing.expectError(error.NotPairwiseCoprime, chineseRemainderTheorem(&rem, &mod));
}

test "crt: invalid modulus returns error" {
    const rem = [_]i64{1};
    const mod = [_]i64{0};
    try testing.expectError(error.InvalidModulus, chineseRemainderTheorem(&rem, &mod));
}

test "crt: length mismatch returns error" {
    const rem = [_]i64{ 1, 2 };
    const mod = [_]i64{3};
    try testing.expectError(error.LengthMismatch, chineseRemainderTheorem(&rem, &mod));
}
