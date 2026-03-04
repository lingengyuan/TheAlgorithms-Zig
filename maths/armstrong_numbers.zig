//! Armstrong Numbers - Zig implementation
//! Reference: https://github.com/TheAlgorithms/Python/blob/master/maths/special_numbers/armstrong_numbers.py

const std = @import("std");
const testing = std.testing;

/// Returns true when `n` is an Armstrong number.
/// Time complexity: O(d), Space complexity: O(1), where d is number of digits.
pub fn armstrongNumber(n: i64) bool {
    if (n < 1) return false;
    const value: u128 = @intCast(n);
    const exponent = digitCount(value);

    var temp = value;
    var total: u128 = 0;
    while (temp > 0) {
        const digit: u8 = @intCast(temp % 10);
        total += powDigit(digit, exponent);
        temp /= 10;
    }
    return total == value;
}

/// Returns true when `n` is a pluperfect number.
/// Time complexity: O(d), Space complexity: O(1)
pub fn pluperfectNumber(n: i64) bool {
    if (n < 1) return false;
    const value: u128 = @intCast(n);
    const exponent = digitCount(value);

    var histogram = [_]u8{0} ** 10;
    var temp = value;
    while (temp > 0) {
        const digit: usize = @intCast(temp % 10);
        histogram[digit] += 1;
        temp /= 10;
    }

    var total: u128 = 0;
    for (histogram, 0..) |count, idx| {
        total += @as(u128, count) * powDigit(@intCast(idx), exponent);
    }
    return total == value;
}

/// Returns true when `n` is a narcissistic number.
/// Time complexity: O(d), Space complexity: O(1)
pub fn narcissisticNumber(n: i64) bool {
    return armstrongNumber(n);
}

fn digitCount(value: u128) u8 {
    var temp = value;
    var count: u8 = 0;
    while (temp > 0) {
        count += 1;
        temp /= 10;
    }
    return count;
}

fn powDigit(base: u8, exponent: u8) u128 {
    var out: u128 = 1;
    var i: u8 = 0;
    while (i < exponent) : (i += 1) {
        out *= @as(u128, base);
    }
    return out;
}

test "armstrong numbers: python passing and failing sets" {
    const passing = [_]i64{ 1, 153, 370, 371, 1_634, 24_678_051 };
    for (passing) |value| {
        try testing.expect(armstrongNumber(value));
        try testing.expect(pluperfectNumber(value));
        try testing.expect(narcissisticNumber(value));
    }

    const failing = [_]i64{ -153, -1, 0, 200, 10_000 };
    for (failing) |value| {
        try testing.expect(!armstrongNumber(value));
        try testing.expect(!pluperfectNumber(value));
        try testing.expect(!narcissisticNumber(value));
    }
}

test "armstrong numbers: method consistency on range" {
    var n: i64 = 1;
    while (n <= 100_000) : (n += 1) {
        const expected = armstrongNumber(n);
        try testing.expectEqual(expected, pluperfectNumber(n));
        try testing.expectEqual(expected, narcissisticNumber(n));
    }
}

test "armstrong numbers: extreme boundary value" {
    try testing.expect(!armstrongNumber(std.math.maxInt(i64)));
}
