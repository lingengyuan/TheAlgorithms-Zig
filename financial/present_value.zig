//! Present Value - Zig implementation
//! Reference: https://github.com/TheAlgorithms/Python/blob/master/financial/present_value.py

const std = @import("std");
const testing = std.testing;

pub const PresentValueError = error{
    NegativeDiscountRate,
    EmptyCashFlows,
};

fn roundTo2(value: f64) f64 {
    return @round(value * 100.0) / 100.0;
}

/// Computes present value of a cash flow stream:
/// sum(cash_flow[i] / (1 + discount_rate)^i), rounded to 2 decimals.
///
/// Time complexity: O(n)
/// Space complexity: O(1)
pub fn presentValue(discount_rate: f64, cash_flows: []const f64) PresentValueError!f64 {
    if (discount_rate < 0) return PresentValueError.NegativeDiscountRate;
    if (cash_flows.len == 0) return PresentValueError.EmptyCashFlows;

    var present_value: f64 = 0.0;
    for (cash_flows, 0..) |cash_flow, i| {
        const year: f64 = @floatFromInt(i);
        present_value += cash_flow / std.math.pow(f64, 1.0 + discount_rate, year);
    }
    return roundTo2(present_value);
}

test "present value: python examples" {
    try testing.expectApproxEqAbs(@as(f64, 4.69), try presentValue(0.13, &[_]f64{ 10, 20.70, -293, 297 }), 1e-12);
    try testing.expectApproxEqAbs(@as(f64, -42739.63), try presentValue(0.07, &[_]f64{ -109129.39, 30923.23, 15098.93, 29734, 39 }), 1e-9);
    try testing.expectApproxEqAbs(@as(f64, 175519.15), try presentValue(0.07, &[_]f64{ 109129.39, 30923.23, 15098.93, 29734, 39 }), 1e-9);
    try testing.expectError(PresentValueError.NegativeDiscountRate, presentValue(-1, &[_]f64{ 109129.39, 30923.23 }));
    try testing.expectError(PresentValueError.EmptyCashFlows, presentValue(0.03, &[_]f64{}));
}

test "present value: boundary and extreme values" {
    try testing.expectApproxEqAbs(@as(f64, 1000.0), try presentValue(0.0, &[_]f64{ 100, 200, 300, 400 }), 1e-12);

    const long = [_]f64{1e8} ** 30;
    const pv_long = try presentValue(0.01, &long);
    try testing.expect(std.math.isFinite(pv_long));
    try testing.expect(pv_long > 0);
}
