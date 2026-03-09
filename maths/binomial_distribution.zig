//! Binomial Distribution - Zig implementation
//! Reference: https://github.com/TheAlgorithms/Python/blob/master/maths/binomial_distribution.py

const std = @import("std");
const testing = std.testing;

pub const BinomialDistributionError = error{
    InvalidInput,
    InvalidProbability,
};

fn factorial(n: i64) u128 {
    var result: u128 = 1;
    var i: i64 = 2;
    while (i <= n) : (i += 1) {
        result *= @as(u128, @intCast(i));
    }
    return result;
}

/// Returns the binomial probability of `successes` out of `trials`.
pub fn binomialDistribution(successes: i64, trials: i64, prob: f64) BinomialDistributionError!f64 {
    if (successes > trials or trials < 0 or successes < 0) return error.InvalidInput;
    if (!(0.0 < prob and prob < 1.0)) return error.InvalidProbability;

    const probability = std.math.pow(f64, prob, @floatFromInt(successes)) *
        std.math.pow(f64, 1.0 - prob, @floatFromInt(trials - successes));
    const coefficient =
        @as(f64, @floatFromInt(factorial(trials))) /
        @as(f64, @floatFromInt(factorial(successes) * factorial(trials - successes)));
    return probability * coefficient;
}

test "binomial distribution: python reference examples" {
    try testing.expectApproxEqAbs(@as(f64, 0.30870000000000003), try binomialDistribution(3, 5, 0.7), 1e-12);
    try testing.expectApproxEqAbs(@as(f64, 0.375), try binomialDistribution(2, 4, 0.5), 1e-12);
}

test "binomial distribution: edge cases" {
    try testing.expectError(error.InvalidInput, binomialDistribution(5, 3, 0.7));
    try testing.expectError(error.InvalidInput, binomialDistribution(-1, 4, 0.5));
    try testing.expectError(error.InvalidProbability, binomialDistribution(2, 4, 1.0));
}
