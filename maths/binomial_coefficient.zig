//! Binomial Coefficient - Zig implementation
//! Reference: https://github.com/TheAlgorithms/Python/blob/master/maths/binomial_coefficient.py

const std = @import("std");
const testing = std.testing;

/// Computes C(n, k) ("n choose k").
/// Returns 0 when k > n.
/// Time complexity: O(min(k, n-k)), Space complexity: O(1)
pub fn binomialCoefficient(n: u64, k: u64) !u64 {
    if (k > n) return 0;
    const kk = if (k < n - k) k else n - k;
    var result: u128 = 1;
    var i: u64 = 1;
    while (i <= kk) : (i += 1) {
        const numerator: u128 = n - kk + i;
        const mul = @mulWithOverflow(result, numerator);
        if (mul[1] != 0) return error.Overflow;
        result = @divExact(mul[0], i);
    }
    if (result > std.math.maxInt(u64)) return error.Overflow;
    return @intCast(result);
}

test "binomial: basic values" {
    try testing.expectEqual(@as(u64, 10), try binomialCoefficient(5, 2));
    try testing.expectEqual(@as(u64, 1), try binomialCoefficient(5, 0));
    try testing.expectEqual(@as(u64, 1), try binomialCoefficient(5, 5));
}

test "binomial: symmetric property" {
    try testing.expectEqual(try binomialCoefficient(20, 3), try binomialCoefficient(20, 17));
}

test "binomial: larger value" {
    try testing.expectEqual(@as(u64, 2_598_960), try binomialCoefficient(52, 5));
}

test "binomial: k greater than n" {
    try testing.expectEqual(@as(u64, 0), try binomialCoefficient(3, 5));
}
