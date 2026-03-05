//! IC 555 Timer - Zig implementation
//! Reference: https://github.com/TheAlgorithms/Python/blob/master/electronics/ic_555_timer.py

const std = @import("std");
const testing = std.testing;

pub const Ic555Error = error{
    NonPositiveValue,
};

/// Computes astable 555 timer frequency in Hz:
/// f = 1.44 / ((R1 + 2*R2) * C_uF) * 10^6.
///
/// Time complexity: O(1)
/// Space complexity: O(1)
pub fn astableFrequency(
    resistance_1: f64,
    resistance_2: f64,
    capacitance: f64,
) Ic555Error!f64 {
    if (resistance_1 <= 0 or resistance_2 <= 0 or capacitance <= 0) {
        return Ic555Error.NonPositiveValue;
    }
    return (1.44 / ((resistance_1 + 2.0 * resistance_2) * capacitance)) * 1e6;
}

/// Computes astable 555 timer duty cycle in percent:
/// duty = (R1 + R2) / (R1 + 2*R2) * 100.
///
/// Time complexity: O(1)
/// Space complexity: O(1)
pub fn astableDutyCycle(resistance_1: f64, resistance_2: f64) Ic555Error!f64 {
    if (resistance_1 <= 0 or resistance_2 <= 0) {
        return Ic555Error.NonPositiveValue;
    }
    return (resistance_1 + resistance_2) / (resistance_1 + 2.0 * resistance_2) * 100.0;
}

test "ic 555 timer: python examples" {
    try testing.expectApproxEqAbs(@as(f64, 1523.8095238095239), try astableFrequency(45, 45, 7), 1e-12);
    try testing.expectApproxEqAbs(@as(f64, 1.7905459175553078), try astableFrequency(356, 234, 976), 1e-12);

    try testing.expectApproxEqAbs(@as(f64, 66.66666666666666), try astableDutyCycle(45, 45), 1e-12);
    try testing.expectApproxEqAbs(@as(f64, 71.60194174757282), try astableDutyCycle(356, 234), 1e-12);
}

test "ic 555 timer: validation and extreme values" {
    try testing.expectError(Ic555Error.NonPositiveValue, astableFrequency(2, -1, 2));
    try testing.expectError(Ic555Error.NonPositiveValue, astableFrequency(45, 45, 0));
    try testing.expectError(Ic555Error.NonPositiveValue, astableDutyCycle(2, -1));
    try testing.expectError(Ic555Error.NonPositiveValue, astableDutyCycle(0, 0));

    const extreme_freq = try astableFrequency(1e-9, 1e-9, 1e-12);
    try testing.expect(std.math.isFinite(extreme_freq));
    try testing.expect(extreme_freq > 0);

    const near_limit_duty = try astableDutyCycle(1e9, 1e-9);
    try testing.expect(near_limit_duty <= 100.0);
    try testing.expect(near_limit_duty > 99.999999999);
}
