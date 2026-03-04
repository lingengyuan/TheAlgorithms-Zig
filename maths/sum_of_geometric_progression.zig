//! Sum of Geometric Progression - Zig implementation
//! Reference: https://github.com/TheAlgorithms/Python/blob/master/maths/sum_of_geometric_progression.py

const std = @import("std");
const testing = std.testing;

/// Returns sum of geometric progression.
/// Time complexity: O(log n) due to exponentiation, Space complexity: O(1)
pub fn sumOfGeometricProgression(first_term: f64, common_ratio: f64, num_of_terms: f64) f64 {
    if (common_ratio == 1.0) return num_of_terms * first_term;
    return (first_term / (1.0 - common_ratio)) * (1.0 - std.math.pow(f64, common_ratio, num_of_terms));
}

test "sum of geometric progression: python reference examples" {
    try testing.expectApproxEqAbs(@as(f64, 1023.0), sumOfGeometricProgression(1, 2, 10), 1e-12);
    try testing.expectApproxEqAbs(@as(f64, 11111.0), sumOfGeometricProgression(1, 10, 5), 1e-12);
    try testing.expectApproxEqAbs(@as(f64, 0.0), sumOfGeometricProgression(0, 2, 10), 1e-12);
    try testing.expectApproxEqAbs(@as(f64, 1.0), sumOfGeometricProgression(1, 0, 10), 1e-12);
    try testing.expectApproxEqAbs(@as(f64, -0.0), sumOfGeometricProgression(1, 2, 0), 1e-12);
    try testing.expectApproxEqAbs(@as(f64, -1023.0), sumOfGeometricProgression(-1, 2, 10), 1e-12);
    try testing.expectApproxEqAbs(@as(f64, -341.0), sumOfGeometricProgression(1, -2, 10), 1e-12);
    try testing.expectApproxEqAbs(@as(f64, -0.9990234375), sumOfGeometricProgression(1, 2, -10), 1e-12);
}

test "sum of geometric progression: edge and extreme cases" {
    try testing.expectApproxEqAbs(@as(f64, 5.0), sumOfGeometricProgression(5, 1, 1), 1e-12);
    try testing.expect(sumOfGeometricProgression(1, 1.0001, 10_000) > 0.0);
}
