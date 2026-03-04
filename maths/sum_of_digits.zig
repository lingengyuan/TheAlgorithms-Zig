//! Sum of Digits - Zig implementation
//! Reference: https://github.com/TheAlgorithms/Python/blob/master/maths/sum_of_digits.py

const std = @import("std");
const testing = std.testing;

/// Returns the sum of decimal digits of `n`.
/// Time complexity: O(d), Space complexity: O(1), where d is number of digits.
pub fn sumOfDigits(n: i64) u64 {
    return sumOfDigitsUnsigned(absAsU128(n));
}

/// Returns the sum of decimal digits of `n` using recursion.
/// Time complexity: O(d), Space complexity: O(d) recursion depth.
pub fn sumOfDigitsRecursion(n: i64) u64 {
    return sumOfDigitsRecursionUnsigned(absAsU128(n));
}

/// Returns the sum of decimal digits using decimal string traversal.
/// Time complexity: O(d), Space complexity: O(d)
pub fn sumOfDigitsCompact(n: i64) u64 {
    const abs_number = absAsU128(n);
    var buffer: [40]u8 = undefined;
    const repr = std.fmt.bufPrint(&buffer, "{d}", .{abs_number}) catch unreachable;

    var total: u64 = 0;
    for (repr) |ch| {
        total += ch - '0';
    }
    return total;
}

fn sumOfDigitsUnsigned(value: u128) u64 {
    var n = value;
    var total: u64 = 0;
    while (n > 0) {
        total += @intCast(n % 10);
        n /= 10;
    }
    return total;
}

fn sumOfDigitsRecursionUnsigned(value: u128) u64 {
    if (value < 10) return @intCast(value);
    return @as(u64, @intCast(value % 10)) + sumOfDigitsRecursionUnsigned(value / 10);
}

fn absAsU128(n: i64) u128 {
    const extended: i128 = n;
    if (extended >= 0) return @intCast(extended);
    return @intCast(-extended);
}

test "sum of digits: python reference examples" {
    try testing.expectEqual(@as(u64, 15), sumOfDigits(12_345));
    try testing.expectEqual(@as(u64, 6), sumOfDigits(123));
    try testing.expectEqual(@as(u64, 6), sumOfDigits(-123));
    try testing.expectEqual(@as(u64, 0), sumOfDigits(0));

    try testing.expectEqual(@as(u64, 15), sumOfDigitsRecursion(12_345));
    try testing.expectEqual(@as(u64, 6), sumOfDigitsRecursion(123));
    try testing.expectEqual(@as(u64, 6), sumOfDigitsRecursion(-123));
    try testing.expectEqual(@as(u64, 0), sumOfDigitsRecursion(0));

    try testing.expectEqual(@as(u64, 15), sumOfDigitsCompact(12_345));
    try testing.expectEqual(@as(u64, 6), sumOfDigitsCompact(123));
    try testing.expectEqual(@as(u64, 6), sumOfDigitsCompact(-123));
    try testing.expectEqual(@as(u64, 0), sumOfDigitsCompact(0));
}

test "sum of digits: implementation consistency" {
    const values = [_]i64{ -999_999, -1, 0, 1, 9, 10, 101, 40_001, 9_876_543_210 };
    for (values) |value| {
        const expected = sumOfDigits(value);
        try testing.expectEqual(expected, sumOfDigitsRecursion(value));
        try testing.expectEqual(expected, sumOfDigitsCompact(value));
    }
}

test "sum of digits: extreme integer boundaries" {
    try testing.expectEqual(@as(u64, 88), sumOfDigits(std.math.maxInt(i64)));
    try testing.expectEqual(@as(u64, 89), sumOfDigits(std.math.minInt(i64)));
}
