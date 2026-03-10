//! Project Euler Problem 686: Powers of Two - Zig implementation
//! Reference: https://github.com/TheAlgorithms/Python/blob/master/project_euler/problem_686/sol1.py

const std = @import("std");
const testing = std.testing;

fn logDifference(number: u64) f64 {
    const log_number = std.math.log10(@as(f64, 2.0)) * @as(f64, @floatFromInt(number));
    const difference = log_number - @floor(log_number);
    return @round(difference * 1_000_000_000_000_000.0) / 1_000_000_000_000_000.0;
}

/// Returns the `position`-th exponent `j` such that `2^j` begins with `123`.
/// Time complexity: O(position)
/// Space complexity: O(1)
pub fn solution(position: u64) u64 {
    if (position == 0) return 0;

    const lower_limit = std.math.log10(@as(f64, 1.23));
    const upper_limit = std.math.log10(@as(f64, 1.24));

    var power: u64 = 90;
    var found: u64 = 0;
    var previous_power: u64 = 0;

    while (found < position) {
        const difference = logDifference(power);
        if (difference >= upper_limit) {
            power += 93;
        } else if (difference < lower_limit) {
            power += 196;
        } else {
            previous_power = power;
            power += 196;
            found += 1;
        }
    }

    return previous_power;
}

test "problem 686: python reference" {
    try testing.expectEqual(@as(u64, 284168), solution(1000));
    try testing.expectEqual(@as(u64, 15924915), solution(56000));
    try testing.expectEqual(@as(u64, 193060223), solution(678910));
}

test "problem 686: logarithmic helper and small positions" {
    try testing.expectApproxEqAbs(@as(f64, 0.092699609758302), logDifference(90), 1e-14);
    try testing.expectApproxEqAbs(@as(f64, 0.090368356648852), logDifference(379), 3e-14);
    try testing.expectEqual(@as(u64, 90), solution(1));
    try testing.expectEqual(@as(u64, 2515), solution(10));
}
