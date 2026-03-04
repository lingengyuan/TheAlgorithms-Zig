//! Decimal Isolate - Zig implementation
//! Reference: https://github.com/TheAlgorithms/Python/blob/master/maths/decimal_isolate.py

const std = @import("std");
const testing = std.testing;

/// Isolates decimal part of `number`.
/// If `digit_amount > 0`, rounds decimal part to that many places.
/// Time complexity: O(1), Space complexity: O(1)
pub fn decimalIsolate(number: f64, digit_amount: i64) f64 {
    const integer_part = @trunc(number); // Python int() on float truncates toward zero
    const decimal_part = number - integer_part;
    if (digit_amount > 0) {
        const digits: u8 = @intCast(@min(digit_amount, 18));
        return roundToDigits(decimal_part, digits);
    }
    return decimal_part;
}

fn roundToDigits(value: f64, digits: u8) f64 {
    var scale: f64 = 1.0;
    var i: u8 = 0;
    while (i < digits) : (i += 1) scale *= 10.0;
    return std.math.round(value * scale) / scale;
}

test "decimal isolate: python reference examples" {
    try testing.expectApproxEqAbs(@as(f64, 0.53), decimalIsolate(1.53, 0), 1e-12);
    try testing.expectApproxEqAbs(@as(f64, 0.3), decimalIsolate(35.345, 1), 1e-12);
    try testing.expectApproxEqAbs(@as(f64, 0.34), decimalIsolate(35.345, 2), 1e-12);
    try testing.expectApproxEqAbs(@as(f64, 0.345), decimalIsolate(35.345, 3), 1e-12);
    try testing.expectApproxEqAbs(@as(f64, -0.789), decimalIsolate(-14.789, 3), 1e-12);
    try testing.expectApproxEqAbs(@as(f64, 0.0), decimalIsolate(0, 2), 1e-12);
    try testing.expectApproxEqAbs(@as(f64, -0.1), decimalIsolate(-14.123, 1), 1e-12);
    try testing.expectApproxEqAbs(@as(f64, -0.12), decimalIsolate(-14.123, 2), 1e-12);
    try testing.expectApproxEqAbs(@as(f64, -0.123), decimalIsolate(-14.123, 3), 1e-12);
}

test "decimal isolate: edge and extreme cases" {
    try testing.expectApproxEqAbs(@as(f64, 0.0), decimalIsolate(123.0, 6), 1e-12);
    try testing.expectApproxEqAbs(@as(f64, -0.999999), decimalIsolate(-1.9999994, 6), 1e-12);
    try testing.expectApproxEqAbs(@as(f64, 0.000001), decimalIsolate(1.000001, 6), 1e-12);
}
