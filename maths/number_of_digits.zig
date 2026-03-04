//! Number of Digits - Zig implementation
//! Reference: https://github.com/TheAlgorithms/Python/blob/master/maths/number_of_digits.py

const std = @import("std");
const testing = std.testing;

/// Returns the number of decimal digits in `n` via repeated division.
/// Time complexity: O(d), Space complexity: O(1)
pub fn numDigits(n: i64) u32 {
    var value = absAsU128(n);
    var digits: u32 = 0;

    while (true) {
        value /= 10;
        digits += 1;
        if (value == 0) break;
    }

    return digits;
}

/// Returns the number of decimal digits in `n` using base-10 logarithm.
/// Time complexity: O(1), Space complexity: O(1)
pub fn numDigitsFast(n: i64) u32 {
    const value = absAsU128(n);
    if (value == 0) return 1;

    const as_float: f64 = @floatFromInt(value);
    var digits: u32 = @as(u32, @intFromFloat(@floor(std.math.log10(as_float)))) + 1;

    // Correct potential floating-point rounding near powers of ten.
    while (digits > 1 and value < pow10(digits - 1)) {
        digits -= 1;
    }
    while (value >= pow10(digits)) {
        digits += 1;
    }

    return digits;
}

/// Returns the number of decimal digits in `n` using decimal-string length.
/// Time complexity: O(d), Space complexity: O(d)
pub fn numDigitsFaster(n: i64) u32 {
    const value = absAsU128(n);
    var buffer: [40]u8 = undefined;
    const repr = std.fmt.bufPrint(&buffer, "{d}", .{value}) catch unreachable;
    return @intCast(repr.len);
}

fn absAsU128(n: i64) u128 {
    const extended: i128 = n;
    if (extended >= 0) return @intCast(extended);
    return @intCast(-extended);
}

fn pow10(exp: u32) u128 {
    var out: u128 = 1;
    var i: u32 = 0;
    while (i < exp) : (i += 1) {
        out *= 10;
    }
    return out;
}

test "number of digits: python reference examples" {
    try testing.expectEqual(@as(u32, 5), numDigits(12_345));
    try testing.expectEqual(@as(u32, 3), numDigits(123));
    try testing.expectEqual(@as(u32, 1), numDigits(0));
    try testing.expectEqual(@as(u32, 1), numDigits(-1));
    try testing.expectEqual(@as(u32, 6), numDigits(-123_456));

    try testing.expectEqual(@as(u32, 5), numDigitsFast(12_345));
    try testing.expectEqual(@as(u32, 3), numDigitsFast(123));
    try testing.expectEqual(@as(u32, 1), numDigitsFast(0));
    try testing.expectEqual(@as(u32, 1), numDigitsFast(-1));
    try testing.expectEqual(@as(u32, 6), numDigitsFast(-123_456));

    try testing.expectEqual(@as(u32, 5), numDigitsFaster(12_345));
    try testing.expectEqual(@as(u32, 3), numDigitsFaster(123));
    try testing.expectEqual(@as(u32, 1), numDigitsFaster(0));
    try testing.expectEqual(@as(u32, 1), numDigitsFaster(-1));
    try testing.expectEqual(@as(u32, 6), numDigitsFaster(-123_456));
}

test "number of digits: methods stay consistent around powers of ten" {
    const values = [_]i64{ 9, 10, 11, 99, 100, 101, 999, 1_000, 1_001, 999_999_999_999_999_999, 1_000_000_000_000_000_000 };
    for (values) |value| {
        const expected = numDigits(value);
        try testing.expectEqual(expected, numDigitsFast(value));
        try testing.expectEqual(expected, numDigitsFaster(value));
        try testing.expectEqual(expected, numDigits(-value));
    }
}

test "number of digits: extreme integer boundaries" {
    try testing.expectEqual(@as(u32, 19), numDigits(std.math.maxInt(i64)));
    try testing.expectEqual(@as(u32, 19), numDigits(std.math.minInt(i64)));
    try testing.expectEqual(@as(u32, 19), numDigitsFast(std.math.maxInt(i64)));
    try testing.expectEqual(@as(u32, 19), numDigitsFast(std.math.minInt(i64)));
    try testing.expectEqual(@as(u32, 19), numDigitsFaster(std.math.maxInt(i64)));
    try testing.expectEqual(@as(u32, 19), numDigitsFaster(std.math.minInt(i64)));
}
