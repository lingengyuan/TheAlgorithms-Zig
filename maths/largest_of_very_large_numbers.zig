//! Largest of Very Large Numbers Helper - Zig implementation
//! Reference: https://github.com/TheAlgorithms/Python/blob/master/maths/largest_of_very_large_numbers.py

const std = @import("std");
const testing = std.testing;

pub const VeryLargeError = error{ExpectedPositiveInput};

/// Reduces x^y to y * log10(x) for comparison of very large powers.
/// Time complexity: O(1), Space complexity: O(1)
pub fn res(x: f64, y: f64) VeryLargeError!f64 {
    if (x != 0.0 and y != 0.0) {
        if (x <= 0.0) return VeryLargeError.ExpectedPositiveInput;
        return y * std.math.log10(x);
    } else if (x == 0.0) {
        return 0.0;
    } else if (y == 0.0) {
        return 0.0;
    }
    unreachable;
}

test "largest of very large numbers: python reference examples" {
    try testing.expectApproxEqAbs(@as(f64, 4.892790030352132), try res(5, 7), 1e-12);
    try testing.expectApproxEqAbs(@as(f64, 0.0), try res(0, 5), 1e-12);
    try testing.expectApproxEqAbs(@as(f64, 0.0), try res(3, 0), 1e-12);
}

test "largest of very large numbers: edge and extreme cases" {
    try testing.expectError(VeryLargeError.ExpectedPositiveInput, res(-1, 5));
    try testing.expectApproxEqAbs(@as(f64, 0.0), try res(0, 0), 1e-12);
    try testing.expect((try res(1e100, 1e6)) > 0.0);
}
