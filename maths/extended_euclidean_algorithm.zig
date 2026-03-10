//! Extended Euclidean Algorithm - Zig implementation
//! Reference: https://github.com/TheAlgorithms/Python/blob/master/maths/extended_euclidean_algorithm.py

const std = @import("std");
const testing = std.testing;

pub const BezoutCoefficients = struct {
    x: i64,
    y: i64,
};

/// Returns coefficients `(x, y)` such that `a*x + b*y = gcd(a, b)`.
/// The behavior follows the Python reference's sign handling.
/// Time complexity: O(log(min(|a|, |b|))), Space complexity: O(1)
pub fn extendedEuclideanAlgorithm(a: i64, b: i64) BezoutCoefficients {
    if (@abs(a) == 1) return .{ .x = a, .y = 0 };
    if (@abs(b) == 1) return .{ .x = 0, .y = b };

    var old_remainder: i128 = a;
    var remainder: i128 = b;
    var old_coeff_a: i128 = 1;
    var coeff_a: i128 = 0;
    var old_coeff_b: i128 = 0;
    var coeff_b: i128 = 1;

    while (remainder != 0) {
        const quotient = @divTrunc(old_remainder, remainder);
        const next_remainder = old_remainder - quotient * remainder;
        old_remainder = remainder;
        remainder = next_remainder;

        const next_coeff_a = old_coeff_a - quotient * coeff_a;
        old_coeff_a = coeff_a;
        coeff_a = next_coeff_a;

        const next_coeff_b = old_coeff_b - quotient * coeff_b;
        old_coeff_b = coeff_b;
        coeff_b = next_coeff_b;
    }

    if (a < 0) old_coeff_a = -old_coeff_a;
    if (b < 0) old_coeff_b = -old_coeff_b;

    return .{
        .x = @intCast(old_coeff_a),
        .y = @intCast(old_coeff_b),
    };
}

test "extended euclidean algorithm: python reference examples" {
    const r1 = extendedEuclideanAlgorithm(1, 24);
    try testing.expectEqual(BezoutCoefficients{ .x = 1, .y = 0 }, r1);

    const r2 = extendedEuclideanAlgorithm(8, 14);
    try testing.expectEqual(BezoutCoefficients{ .x = 2, .y = -1 }, r2);

    const r3 = extendedEuclideanAlgorithm(240, 46);
    try testing.expectEqual(BezoutCoefficients{ .x = -9, .y = 47 }, r3);
}

test "extended euclidean algorithm: sign and zero edge cases" {
    try testing.expectEqual(BezoutCoefficients{ .x = 1, .y = 0 }, extendedEuclideanAlgorithm(1, -4));
    try testing.expectEqual(BezoutCoefficients{ .x = -1, .y = 0 }, extendedEuclideanAlgorithm(-2, -4));
    try testing.expectEqual(BezoutCoefficients{ .x = 0, .y = -1 }, extendedEuclideanAlgorithm(0, -4));
    try testing.expectEqual(BezoutCoefficients{ .x = 1, .y = 0 }, extendedEuclideanAlgorithm(2, 0));
}

