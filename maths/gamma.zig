//! Gamma Function - Zig implementation
//! Reference: https://github.com/TheAlgorithms/Python/blob/master/maths/gamma.py

const std = @import("std");
const testing = std.testing;

pub const GammaError = error{
    MathDomain,
    MathRange,
    UnsupportedValue,
};

const lanczos_g = 7.0;
const sqrt_two_pi = 2.5066282746310005024;
const lanczos_coeffs = [_]f64{
    0.99999999999980993,
    676.5203681218851,
    -1259.1392167224028,
    771.32342877765313,
    -176.61502916214059,
    12.507343278686905,
    -0.13857109526572012,
    9.9843695780195716e-6,
    1.5056327351493116e-7,
};

/// Approximates `Gamma(z)` for positive real values using the Lanczos approximation.
/// Time complexity: O(1), Space complexity: O(1)
pub fn gammaIterative(num: f64) GammaError!f64 {
    if (!(num > 0.0)) return error.MathDomain;
    return lanczosGamma(num);
}

/// Computes `Gamma(z)` recursively for positive integers and half-integers.
/// Mirrors the restricted-domain behavior of the Python reference.
/// Time complexity: O(num), Space complexity: O(num)
pub fn gammaRecursive(num: f64) GammaError!f64 {
    if (!(num > 0.0)) return error.MathDomain;
    if (num > 171.5) return error.MathRange;

    const doubled = num * 2.0;
    const rounded = @round(doubled);
    if (@abs(doubled - rounded) > 1e-12) return error.UnsupportedValue;

    if (@abs(num - 0.5) <= 1e-12) return @sqrt(std.math.pi);
    if (@abs(num - 1.0) <= 1e-12) return 1.0;
    return (num - 1.0) * try gammaRecursive(num - 1.0);
}

fn lanczosGamma(z: f64) f64 {
    if (z < 0.5) {
        return std.math.pi / (@sin(std.math.pi * z) * lanczosGamma(1.0 - z));
    }

    const shifted = z - 1.0;
    var sum = lanczos_coeffs[0];
    for (lanczos_coeffs[1..], 1..) |coeff, index| {
        sum += coeff / (shifted + @as(f64, @floatFromInt(index)));
    }

    const t = shifted + lanczos_g + 0.5;
    return sqrt_two_pi * std.math.pow(f64, t, shifted + 0.5) * @exp(-t) * sum;
}

test "gamma: iterative matches reference values" {
    try testing.expectApproxEqAbs(@as(f64, 40320.0), try gammaIterative(9.0), 1e-7);
    try testing.expectApproxEqAbs(@as(f64, 2.6834373819557675), try gammaIterative(3.3), 1e-12);
    try testing.expectApproxEqAbs(@as(f64, 1.7724538509055159), try gammaIterative(0.5), 1e-12);
}

test "gamma: recursive matches integer and half integer values" {
    try testing.expectApproxEqAbs(@as(f64, 1.0), try gammaRecursive(1.0), 1e-12);
    try testing.expectApproxEqAbs(@as(f64, 1.0), try gammaRecursive(2.0), 1e-12);
    try testing.expectApproxEqAbs(@as(f64, 3.323350970447842), try gammaRecursive(3.5), 1e-12);
    try testing.expectApproxEqRel(@as(f64, 9.483367566824801e+307), try gammaRecursive(171.5), 1e-12);
}

test "gamma: invalid and unsupported inputs" {
    try testing.expectError(error.MathDomain, gammaIterative(0.0));
    try testing.expectError(error.MathDomain, gammaRecursive(-1.1));
    try testing.expectError(error.UnsupportedValue, gammaRecursive(1.1));
    try testing.expectError(error.MathRange, gammaRecursive(172.0));
}

test "gamma: extreme small positive input remains finite" {
    const value = try gammaIterative(1e-12);
    try testing.expect(std.math.isFinite(value));
    try testing.expect(value > 1e11);
}
