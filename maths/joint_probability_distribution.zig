//! Joint Probability Distribution - Zig implementation
//! Reference: https://github.com/TheAlgorithms/Python/blob/master/maths/joint_probability_distribution.py

const std = @import("std");
const testing = std.testing;

pub const JointProbability = struct {
    x: i64,
    y: i64,
    probability: f64,
};

/// Computes the Cartesian-product joint distribution.
/// Caller owns the returned slice.
/// Time complexity: O(|X| * |Y|), Space complexity: O(|X| * |Y|)
pub fn jointProbabilityDistribution(
    allocator: std.mem.Allocator,
    x_values: []const i64,
    y_values: []const i64,
    x_probabilities: []const f64,
    y_probabilities: []const f64,
) ![]JointProbability {
    var out = std.ArrayListUnmanaged(JointProbability){};
    defer out.deinit(allocator);

    for (x_values, x_probabilities) |x, px| {
        for (y_values, y_probabilities) |y, py| {
            try out.append(allocator, .{
                .x = x,
                .y = y,
                .probability = px * py,
            });
        }
    }
    return out.toOwnedSlice(allocator);
}

/// Returns the expectation of `values` under `probabilities`.
pub fn expectation(values: []const i64, probabilities: []const f64) f64 {
    var total: f64 = 0;
    for (values, probabilities) |value, probability| {
        total += @as(f64, @floatFromInt(value)) * probability;
    }
    return total;
}

/// Returns the variance of `values` under `probabilities`.
pub fn variance(values: []const i64, probabilities: []const f64) f64 {
    const mean = expectation(values, probabilities);
    var total: f64 = 0;
    for (values, probabilities) |value, probability| {
        const diff = @as(f64, @floatFromInt(value)) - mean;
        total += diff * diff * probability;
    }
    return total;
}

/// Returns the covariance of independent variables X and Y.
pub fn covariance(
    x_values: []const i64,
    y_values: []const i64,
    x_probabilities: []const f64,
    y_probabilities: []const f64,
) f64 {
    const mean_x = expectation(x_values, x_probabilities);
    const mean_y = expectation(y_values, y_probabilities);
    var total: f64 = 0;
    for (x_values, x_probabilities) |x, px| {
        for (y_values, y_probabilities) |y, py| {
            total += (@as(f64, @floatFromInt(x)) - mean_x) * (@as(f64, @floatFromInt(y)) - mean_y) * px * py;
        }
    }
    return total;
}

/// Returns the standard deviation from a variance value.
pub fn standardDeviation(value: f64) f64 {
    return @sqrt(value);
}

test "joint probability distribution: python reference examples" {
    const alloc = testing.allocator;
    const distribution = try jointProbabilityDistribution(
        alloc,
        &[_]i64{ 1, 2 },
        &[_]i64{ -2, 5, 8 },
        &[_]f64{ 0.7, 0.3 },
        &[_]f64{ 0.3, 0.5, 0.2 },
    );
    defer alloc.free(distribution);

    try testing.expectEqual(@as(usize, 6), distribution.len);
    try testing.expectApproxEqAbs(@as(f64, 0.14), distribution[2].probability, 1e-12);
    try testing.expectApproxEqAbs(@as(f64, 1.3), expectation(&[_]i64{ 1, 2 }, &[_]f64{ 0.7, 0.3 }), 1e-12);
    try testing.expectApproxEqAbs(@as(f64, 0.21), variance(&[_]i64{ 1, 2 }, &[_]f64{ 0.7, 0.3 }), 1e-12);
}

test "joint probability distribution: edge cases" {
    try testing.expectApproxEqAbs(@as(f64, 0.0), covariance(&[_]i64{ 1, 2 }, &[_]i64{ -2, 5, 8 }, &[_]f64{ 0.7, 0.3 }, &[_]f64{ 0.3, 0.5, 0.2 }), 1e-12);
    try testing.expectApproxEqAbs(@as(f64, 0.458257569495584), standardDeviation(0.21), 1e-12);
}
