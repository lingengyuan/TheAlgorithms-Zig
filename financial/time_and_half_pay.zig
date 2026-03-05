//! Time and Half Pay - Zig implementation
//! Reference: https://github.com/TheAlgorithms/Python/blob/master/financial/time_and_half_pay.py

const std = @import("std");
const testing = std.testing;

/// Computes pay with overtime rule:
/// overtime gets an extra half-rate beyond normal pay.
///
/// Time complexity: O(1)
/// Space complexity: O(1)
pub fn pay(hours_worked: f64, pay_rate: f64, hours: f64) f64 {
    const normal_pay = hours_worked * pay_rate;
    const over_time = @max(0.0, hours_worked - hours);
    const over_time_pay = over_time * pay_rate / 2.0;
    return normal_pay + over_time_pay;
}

/// Convenience helper matching Python default `hours=40`.
pub fn payDefault(hours_worked: f64, pay_rate: f64) f64 {
    return pay(hours_worked, pay_rate, 40.0);
}

test "time and half pay: python examples" {
    try testing.expectApproxEqAbs(@as(f64, 41.5), payDefault(41, 1), 1e-12);
    try testing.expectApproxEqAbs(@as(f64, 1472.5), payDefault(65, 19), 1e-12);
    try testing.expectApproxEqAbs(@as(f64, 10.0), payDefault(10, 1), 1e-12);
}

test "time and half pay: boundary and extreme values" {
    try testing.expectApproxEqAbs(@as(f64, 0.0), payDefault(0, 10), 1e-12);
    try testing.expectApproxEqAbs(@as(f64, 63.75), pay(40, 1.5, 35), 1e-12);

    const huge = payDefault(1e9, 1e6);
    try testing.expect(std.math.isFinite(huge));
    try testing.expect(huge > 0);
}
