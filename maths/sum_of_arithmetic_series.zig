//! Sum of Arithmetic Series - Zig implementation
//! Reference: https://github.com/TheAlgorithms/Python/blob/master/maths/sum_of_arithmetic_series.py

const std = @import("std");
const testing = std.testing;

/// Returns sum of arithmetic progression using closed-form formula.
/// Time complexity: O(1), Space complexity: O(1)
pub fn sumOfSeries(first_term: f64, common_diff: f64, num_of_terms: f64) f64 {
    return (num_of_terms / 2.0) * (2.0 * first_term + (num_of_terms - 1.0) * common_diff);
}

test "sum of arithmetic series: python reference examples" {
    try testing.expectApproxEqAbs(@as(f64, 55.0), sumOfSeries(1, 1, 10), 1e-12);
    try testing.expectApproxEqAbs(@as(f64, 49_600.0), sumOfSeries(1, 10, 100), 1e-12);
}

test "sum of arithmetic series: edge and extreme cases" {
    try testing.expectApproxEqAbs(@as(f64, 0.0), sumOfSeries(1, 1, 0), 1e-12);
    try testing.expectApproxEqAbs(@as(f64, 45.0), sumOfSeries(1, 1, -10), 1e-12);
    try testing.expect(sumOfSeries(1e6, 1e6, 1e6) > 0.0);
}
