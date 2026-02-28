//! Extended Euclidean Algorithm - Zig implementation
//! Reference: https://github.com/TheAlgorithms/Python/blob/master/maths/extended_euclidean_algorithm.py

const std = @import("std");
const testing = std.testing;

pub const ExtendedGcdResult = struct {
    gcd: i64,
    x: i64,
    y: i64,
};

/// Computes gcd(a, b) and coefficients x, y such that: a*x + b*y = gcd(a, b).
/// Ensures returned gcd is non-negative.
/// Time complexity: O(log(min(|a|, |b|))), Space complexity: O(1)
pub fn extendedEuclidean(a: i64, b: i64) ExtendedGcdResult {
    var old_r: i128 = a;
    var r: i128 = b;
    var old_s: i128 = 1;
    var s: i128 = 0;
    var old_t: i128 = 0;
    var t: i128 = 1;

    while (r != 0) {
        const q: i128 = @divTrunc(old_r, r);

        const tmp_r = old_r - q * r;
        old_r = r;
        r = tmp_r;

        const tmp_s = old_s - q * s;
        old_s = s;
        s = tmp_s;

        const tmp_t = old_t - q * t;
        old_t = t;
        t = tmp_t;
    }

    var gcd_i128 = old_r;
    var x_i128 = old_s;
    var y_i128 = old_t;
    if (gcd_i128 < 0) {
        gcd_i128 = -gcd_i128;
        x_i128 = -x_i128;
        y_i128 = -y_i128;
    }

    return .{
        .gcd = @intCast(gcd_i128),
        .x = @intCast(x_i128),
        .y = @intCast(y_i128),
    };
}

test "extended euclidean: basic case" {
    const r = extendedEuclidean(30, 20);
    try testing.expectEqual(@as(i64, 10), r.gcd);
    try testing.expectEqual(@as(i128, 10), @as(i128, 30) * r.x + @as(i128, 20) * r.y);
}

test "extended euclidean: coprime values" {
    const r = extendedEuclidean(35, 64);
    try testing.expectEqual(@as(i64, 1), r.gcd);
    try testing.expectEqual(@as(i128, 1), @as(i128, 35) * r.x + @as(i128, 64) * r.y);
}

test "extended euclidean: with negative input" {
    const r = extendedEuclidean(-25, 10);
    try testing.expectEqual(@as(i64, 5), r.gcd);
    try testing.expectEqual(@as(i128, 5), @as(i128, -25) * r.x + @as(i128, 10) * r.y);
}

test "extended euclidean: one zero" {
    const r = extendedEuclidean(0, 7);
    try testing.expectEqual(@as(i64, 7), r.gcd);
    try testing.expectEqual(@as(i128, 7), @as(i128, 0) * r.x + @as(i128, 7) * r.y);
}

test "extended euclidean: both zero" {
    const r = extendedEuclidean(0, 0);
    try testing.expectEqual(@as(i64, 0), r.gcd);
    try testing.expectEqual(@as(i128, 0), @as(i128, 0) * r.x + @as(i128, 0) * r.y);
}
