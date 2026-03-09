//! Fast Inverse Square Root - Zig implementation
//! Reference: https://github.com/TheAlgorithms/Python/blob/master/maths/fast_inverse_sqrt.py

const std = @import("std");
const testing = std.testing;

pub const FastInverseSqrtError = error{InvalidInput};

/// Computes the Quake III fast inverse square root approximation.
/// Time complexity: O(1), Space complexity: O(1)
pub fn fastInverseSqrt(number: f32) FastInverseSqrtError!f32 {
    if (number <= 0) return error.InvalidInput;
    var i: i32 = @bitCast(number);
    i = 0x5F3759DF - (i >> 1);
    var y: f32 = @bitCast(i);
    y = y * (1.5 - 0.5 * number * y * y);
    return y;
}

test "fast inverse sqrt: python reference examples" {
    try testing.expectApproxEqAbs(@as(f32, 0.3156858), try fastInverseSqrt(10), 1e-6);
    try testing.expectApproxEqAbs(@as(f32, 0.49915358), try fastInverseSqrt(4), 1e-6);
    try testing.expectApproxEqAbs(@as(f32, 0.49328494), try fastInverseSqrt(4.1), 1e-6);
}

test "fast inverse sqrt: edge and extreme cases" {
    try testing.expectError(error.InvalidInput, fastInverseSqrt(0));
    try testing.expectError(error.InvalidInput, fastInverseSqrt(-1));

    const approx = try fastInverseSqrt(64);
    try testing.expect(@abs(approx - 0.125) < 0.002);
}
