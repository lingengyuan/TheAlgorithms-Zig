//! Extended Euclidean Algorithm - Zig implementation
//! Reference: https://github.com/TheAlgorithms/Python/blob/master/maths/extended_euclidean_algorithm.py

const std = @import("std");
const testing = std.testing;

pub const BezoutCoefficients = struct {
    x: i64,
    y: i64,
};

fn absToI128(value: i64) i128 {
    const widened: i128 = value;
    return if (widened < 0) -widened else widened;
}

fn bezoutAbs(a: i128, b: i128) struct { x: i128, y: i128 } {
    if (b == 0) return .{ .x = 1, .y = 0 };
    const next = bezoutAbs(b, @mod(a, b));
    return .{
        .x = next.y,
        .y = next.x - @divTrunc(a, b) * next.y,
    };
}

/// Returns coefficients `(x, y)` such that `a*x + b*y = gcd(a, b)`.
/// Time complexity: O(log(min(|a|, |b|))), Space complexity: O(1)
pub fn extendedEuclideanAlgorithm(a: i64, b: i64) BezoutCoefficients {
    if (a == 0 and b == 0) return .{ .x = 0, .y = 0 };
    if (absToI128(a) == 1) return .{ .x = if (a < 0) -1 else 1, .y = 0 };
    if (absToI128(b) == 1) return .{ .x = 0, .y = if (b < 0) -1 else 1 };

    var coeffs = bezoutAbs(absToI128(a), absToI128(b));
    if (a < 0) coeffs.x = -coeffs.x;
    if (b < 0) coeffs.y = -coeffs.y;

    return .{
        .x = @intCast(coeffs.x),
        .y = @intCast(coeffs.y),
    };
}

fn gcdAbs(a: i64, b: i64) i64 {
    const aa = absToI128(a);
    const bb = absToI128(b);
    var x = aa;
    var y = bb;
    while (y != 0) {
        const rem = @mod(x, y);
        x = y;
        y = rem;
    }
    return @intCast(x);
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

test "extended euclidean algorithm: coefficients satisfy bezout identity for signed inputs" {
    const pairs = [_][2]i64{
        .{ 8, -14 },
        .{ -8, 14 },
        .{ -8, -14 },
        .{ 240, 46 },
        .{ 0, -4 },
    };

    for (pairs) |pair| {
        const result = extendedEuclideanAlgorithm(pair[0], pair[1]);
        const lhs = @as(i128, pair[0]) * result.x + @as(i128, pair[1]) * result.y;
        try testing.expectEqual(@as(i128, gcdAbs(pair[0], pair[1])), lhs);
    }
}
