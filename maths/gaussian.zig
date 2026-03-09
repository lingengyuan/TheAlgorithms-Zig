//! Gaussian Function - Zig implementation
//! Reference: https://github.com/TheAlgorithms/Python/blob/master/maths/gaussian.py

const std = @import("std");
const testing = std.testing;

pub const GaussianError = error{InvalidSigma};

/// Returns the Gaussian function value at `x`.
/// Time complexity: O(1), Space complexity: O(1)
pub fn gaussian(x: f64, mu: f64, sigma: f64) GaussianError!f64 {
    if (sigma == 0.0) return error.InvalidSigma;
    return 1.0 / @sqrt(2.0 * std.math.pi * sigma * sigma) * @exp(-((x - mu) * (x - mu)) / (2.0 * sigma * sigma));
}

test "gaussian: python reference examples" {
    try testing.expectApproxEqAbs(@as(f64, 0.24197072451914337), try gaussian(1, 0, 1), 1e-12);
    try testing.expectApproxEqAbs(@as(f64, 3.342714441794458e-126), try gaussian(24, 0, 1), 1e-138);
    try testing.expectApproxEqAbs(@as(f64, 0.06475879783294587), try gaussian(1, 4, 2), 1e-12);
}

test "gaussian: edge and extreme cases" {
    try testing.expectApproxEqAbs(@as(f64, 0.3989422804014327), try gaussian(0, 0, 1), 1e-12);
    try testing.expectApproxEqAbs(@as(f64, 0.0), try gaussian(2523, 234234, 3425), 1e-12);
    try testing.expectError(error.InvalidSigma, gaussian(1, 0, 0));
}
