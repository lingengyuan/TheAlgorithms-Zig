//! Power (Divide and Conquer) - Zig implementation
//! Reference: https://github.com/TheAlgorithms/Python/blob/master/divide_and_conquer/power.py

const std = @import("std");
const testing = std.testing;

pub const PowerError = error{ DivisionByZero, ExponentOverflow };

fn actualPower(base: f64, exponent: u64) f64 {
    if (exponent == 0) return 1.0;

    const half = actualPower(base, exponent / 2);
    if (exponent % 2 == 0) {
        return half * half;
    }
    return base * half * half;
}

/// Computes a^b by divide and conquer.
/// Returns floating-point output to support negative exponents.
///
/// Time complexity: O(log |b|)
/// Space complexity: O(log |b|) recursion depth
pub fn power(a: i64, b: i64) PowerError!f64 {
    if (b < 0) {
        if (a == 0) return PowerError.DivisionByZero;
        if (b == std.math.minInt(i64)) return PowerError.ExponentOverflow;

        const exp: u64 = @intCast(-b);
        return 1.0 / actualPower(@floatFromInt(a), exp);
    }

    const exp: u64 = @intCast(b);
    return actualPower(@floatFromInt(a), exp);
}

test "power divide and conquer: python examples" {
    try testing.expectApproxEqAbs(@as(f64, 4096.0), try power(4, 6), 1e-12);
    try testing.expectApproxEqAbs(@as(f64, 8.0), try power(2, 3), 1e-12);
    try testing.expectApproxEqAbs(@as(f64, -8.0), try power(-2, 3), 1e-12);
    try testing.expectApproxEqAbs(@as(f64, 0.125), try power(2, -3), 1e-12);
    try testing.expectApproxEqAbs(@as(f64, -0.125), try power(-2, -3), 1e-12);
}

test "power divide and conquer: edge and extreme cases" {
    try testing.expectApproxEqAbs(@as(f64, 1.0), try power(7, 0), 1e-12);
    try testing.expectApproxEqAbs(@as(f64, 1.0), try power(0, 0), 1e-12);
    try testing.expectError(PowerError.DivisionByZero, power(0, -1));
    try testing.expectError(PowerError.ExponentOverflow, power(2, std.math.minInt(i64)));

    const value = try power(2, 60);
    try testing.expect(value > 1e18 and value < 1.2e18);
}
