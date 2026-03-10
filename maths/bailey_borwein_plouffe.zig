//! Bailey-Borwein-Plouffe Pi Digit Extraction - Zig implementation
//! Reference: https://github.com/TheAlgorithms/Python/blob/master/maths/bailey_borwein_plouffe.py

const std = @import("std");
const testing = std.testing;

pub const BbpError = error{
    InvalidDigitPosition,
    InvalidPrecision,
};

/// Returns the hexadecimal digit of pi at the requested 1-based position after the decimal point.
/// Time complexity: O((digit_position + precision) * log digit_position), Space complexity: O(1)
pub fn baileyBorweinPlouffe(digit_position: i64, precision: i64) BbpError!u8 {
    if (digit_position <= 0) return error.InvalidDigitPosition;
    if (precision < 0) return error.InvalidPrecision;

    const position: u64 = @intCast(digit_position);
    const precision_u: u64 = @intCast(precision);

    const sum_result = 4.0 * subsum(position, 1, precision_u) -
        2.0 * subsum(position, 4, precision_u) -
        subsum(position, 5, precision_u) -
        subsum(position, 6, precision_u);

    const fractional = sum_result - @floor(sum_result);
    const digit_index: usize = @intFromFloat(@floor(fractional * 16.0 + 1e-12));
    return "0123456789abcdef"[digit_index];
}

fn subsum(digit_position: u64, denominator_addend: u64, precision: u64) f64 {
    var total: f64 = 0.0;
    var sum_index: u64 = 0;
    while (sum_index < digit_position + precision) : (sum_index += 1) {
        const denominator = 8 * sum_index + denominator_addend;
        const exponential_term = if (sum_index < digit_position)
            @as(f64, @floatFromInt(modPow(16, digit_position - 1 - sum_index, denominator)))
        else
            std.math.pow(f64, 16.0, -@as(f64, @floatFromInt(sum_index - (digit_position - 1))));
        total += exponential_term / @as(f64, @floatFromInt(denominator));
    }
    return total;
}

fn modPow(base: u64, exponent: u64, modulus: u64) u64 {
    var result: u64 = 1 % modulus;
    var current = base % modulus;
    var power = exponent;

    while (power > 0) {
        if (power & 1 == 1) result = @intCast((@as(u128, result) * current) % modulus);
        current = @intCast((@as(u128, current) * current) % modulus);
        power >>= 1;
    }

    return result;
}

test "bailey borwein plouffe: python reference digits" {
    var digits: [10]u8 = undefined;
    for (0..10) |index| {
        digits[index] = try baileyBorweinPlouffe(@intCast(index + 1), 1000);
    }
    try testing.expectEqualStrings("243f6a8885", &digits);
    try testing.expectEqual(@as(u8, '6'), try baileyBorweinPlouffe(5, 10_000));
}

test "bailey borwein plouffe: invalid inputs" {
    try testing.expectError(error.InvalidDigitPosition, baileyBorweinPlouffe(0, 1000));
    try testing.expectError(error.InvalidDigitPosition, baileyBorweinPlouffe(-10, 1000));
    try testing.expectError(error.InvalidPrecision, baileyBorweinPlouffe(2, -10));
}

test "bailey borwein plouffe: extreme precision stability on early digits" {
    try testing.expectEqual(@as(u8, '2'), try baileyBorweinPlouffe(1, 25_000));
    try testing.expectEqual(@as(u8, '8'), try baileyBorweinPlouffe(8, 25_000));
}
