//! Karatsuba Multiplication - Zig implementation
//! Reference: https://github.com/TheAlgorithms/Python/blob/master/maths/karatsuba.py

const std = @import("std");
const testing = std.testing;

fn digitCount(n: u64) u32 {
    if (n < 10) return 1;
    var value = n;
    var count: u32 = 0;
    while (value > 0) : (value /= 10) count += 1;
    return count;
}

/// Multiplies integers using the Karatsuba divide-and-conquer algorithm.
/// Time complexity: O(n^log2(3)), Space complexity: O(log n)
pub fn karatsuba(a: i64, b: i64) i128 {
    const sign: i128 = if ((a < 0) != (b < 0)) -1 else 1;
    const ua: u64 = @intCast(@abs(a));
    const ub: u64 = @intCast(@abs(b));
    return sign * karatsubaUnsigned(ua, ub);
}

fn karatsubaUnsigned(a: u64, b: u64) i128 {
    if (a < 10 or b < 10) return @as(i128, a) * @as(i128, b);

    const m1 = @max(digitCount(a), digitCount(b));
    const m2 = m1 / 2;
    const power = std.math.pow(u64, 10, m2);
    const a1 = a / power;
    const a2 = a % power;
    const b1 = b / power;
    const b2 = b % power;

    const x = karatsubaUnsigned(a2, b2);
    const y = karatsubaUnsigned(a1 + a2, b1 + b2);
    const z = karatsubaUnsigned(a1, b1);
    return z * @as(i128, power) * @as(i128, power) + (y - z - x) * @as(i128, power) + x;
}

test "karatsuba: python reference examples" {
    try testing.expectEqual(@as(i128, 15463 * 23489), karatsuba(15463, 23489));
    try testing.expectEqual(@as(i128, 27), karatsuba(3, 9));
}

test "karatsuba: edge and extreme cases" {
    try testing.expectEqual(@as(i128, 0), karatsuba(0, 999));
    try testing.expectEqual(@as(i128, -56088), karatsuba(-123, 456));
    try testing.expectEqual(@as(i128, 121932631112635269), karatsuba(123456789, 987654321));
}
