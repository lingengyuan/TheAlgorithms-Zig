//! Monte Carlo Estimators - Zig implementation
//! Reference: https://github.com/TheAlgorithms/Python/blob/master/maths/monte_carlo.py

const std = @import("std");
const testing = std.testing;

pub const MonteCarloError = error{InvalidIterationCount};

/// Estimates pi by sampling points in the square `[-1, 1] x [-1, 1]`.
/// Time complexity: O(iterations), Space complexity: O(1)
pub fn piEstimator(iterations: usize, random: std.Random) MonteCarloError!f64 {
    if (iterations == 0) return error.InvalidIterationCount;

    const rng = random;
    var inside: usize = 0;
    for (0..iterations) |_| {
        const x = rng.float(f64) * 2.0 - 1.0;
        const y = rng.float(f64) * 2.0 - 1.0;
        if (x * x + y * y <= 1.0) inside += 1;
    }
    return 4.0 * @as(f64, @floatFromInt(inside)) / @as(f64, @floatFromInt(iterations));
}

/// Estimates the area under a non-negative function on `[min_value, max_value]`.
/// Time complexity: O(iterations), Space complexity: O(1)
pub fn areaUnderCurveEstimator(
    iterations: usize,
    random: std.Random,
    min_value: f64,
    max_value: f64,
    comptime function_to_integrate: fn (f64) f64,
) MonteCarloError!f64 {
    if (iterations == 0) return error.InvalidIterationCount;

    const rng = random;
    var total: f64 = 0.0;
    for (0..iterations) |_| {
        const x = min_value + (max_value - min_value) * rng.float(f64);
        total += function_to_integrate(x);
    }
    return total / @as(f64, @floatFromInt(iterations)) * (max_value - min_value);
}

/// Estimates pi using the area under the semicircle `sqrt(4 - x^2)` on `[0, 2]`.
/// Time complexity: O(iterations), Space complexity: O(1)
pub fn piEstimatorUsingAreaUnderCurve(iterations: usize, random: std.Random) MonteCarloError!f64 {
    return areaUnderCurveEstimator(iterations, random, 0.0, 2.0, semiCircle);
}

fn semiCircle(x: f64) f64 {
    return @sqrt(4.0 - x * x);
}

fn constantTwo(_: f64) f64 {
    return 2.0;
}

fn identity(x: f64) f64 {
    return x;
}

test "monte carlo: pi estimator deterministic with fixed seed" {
    var prng_a = std.Random.DefaultPrng.init(99);
    var prng_b = std.Random.DefaultPrng.init(99);
    try testing.expectEqual(
        try piEstimator(5000, prng_a.random()),
        try piEstimator(5000, prng_b.random()),
    );
}

test "monte carlo: area estimator exact for constant function" {
    var prng = std.Random.DefaultPrng.init(5);
    const area = try areaUnderCurveEstimator(1000, prng.random(), -3.0, 7.0, constantTwo);
    try testing.expectEqual(@as(f64, 20.0), area);
}

test "monte carlo: identity area and pi estimators converge within tolerance" {
    var prng_area = std.Random.DefaultPrng.init(11);
    const identity_area = try areaUnderCurveEstimator(500_000, prng_area.random(), 0.0, 1.0, identity);
    try testing.expect(@abs(identity_area - 0.5) < 0.01);

    var prng_pi = std.Random.DefaultPrng.init(12);
    const pi_square = try piEstimator(500_000, prng_pi.random());
    try testing.expect(@abs(pi_square - std.math.pi) < 0.02);

    var prng_curve = std.Random.DefaultPrng.init(13);
    const pi_curve = try piEstimatorUsingAreaUnderCurve(500_000, prng_curve.random());
    try testing.expect(@abs(pi_curve - std.math.pi) < 0.02);
}

test "monte carlo: invalid and extreme cases" {
    var prng = std.Random.DefaultPrng.init(1);
    try testing.expectError(error.InvalidIterationCount, piEstimator(0, prng.random()));
    try testing.expectError(error.InvalidIterationCount, areaUnderCurveEstimator(0, prng.random(), 0.0, 1.0, identity));

    var prng_single = std.Random.DefaultPrng.init(2);
    const estimate = try piEstimator(1, prng_single.random());
    try testing.expect(estimate == 0.0 or estimate == 4.0);
}
