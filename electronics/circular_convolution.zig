//! Circular Convolution - Zig implementation
//! Reference: https://github.com/TheAlgorithms/Python/blob/master/electronics/circular_convolution.py

const std = @import("std");
const testing = std.testing;

fn roundToTwoDecimals(value: f64) f64 {
    return @round(value * 100.0) / 100.0;
}

/// Computes circular convolution of two signals by zero-padding the shorter
/// signal to max(len(x), len(h)), matching the Python reference behavior.
/// Returned slice is allocator-owned and must be freed by the caller.
///
/// Time complexity: O(n^2)
/// Space complexity: O(n)
pub fn circularConvolution(
    allocator: std.mem.Allocator,
    first_signal: []const f64,
    second_signal: []const f64,
) ![]f64 {
    const n = @max(first_signal.len, second_signal.len);
    const result = try allocator.alloc(f64, n);
    @memset(result, 0.0);

    if (n == 0) {
        return result;
    }

    const first_padded = try allocator.alloc(f64, n);
    defer allocator.free(first_padded);
    @memset(first_padded, 0.0);
    std.mem.copyForwards(f64, first_padded[0..first_signal.len], first_signal);

    const second_padded = try allocator.alloc(f64, n);
    defer allocator.free(second_padded);
    @memset(second_padded, 0.0);
    std.mem.copyForwards(f64, second_padded[0..second_signal.len], second_signal);

    for (0..n) |i| {
        var sum: f64 = 0.0;
        for (0..n) |j| {
            const index = (i + n - j) % n;
            sum += first_padded[j] * second_padded[index];
        }
        result[i] = roundToTwoDecimals(sum);
    }

    return result;
}

test "circular convolution: python examples" {
    const allocator = testing.allocator;

    const first1 = [_]f64{ 2, 1, 2, -1 };
    const second1 = [_]f64{ 1, 2, 3, 4 };
    const result1 = try circularConvolution(allocator, &first1, &second1);
    defer allocator.free(result1);
    try testing.expectEqualSlices(f64, &[_]f64{ 10.0, 10.0, 6.0, 14.0 }, result1);

    const first2 = [_]f64{ 0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6 };
    const second2 = [_]f64{ 0.1, 0.3, 0.5, 0.7, 0.9, 1.1, 1.3, 1.5 };
    const result2 = try circularConvolution(allocator, &first2, &second2);
    defer allocator.free(result2);
    try testing.expectEqualSlices(f64, &[_]f64{ 5.2, 6.0, 6.48, 6.64, 6.48, 6.0, 5.2, 4.08 }, result2);

    const first3 = [_]f64{ -1, 1, 2, -2 };
    const second3 = [_]f64{ 0.5, 1, -1, 2, 0.75 };
    const result3 = try circularConvolution(allocator, &first3, &second3);
    defer allocator.free(result3);
    try testing.expectEqualSlices(f64, &[_]f64{ 6.25, -3.0, 1.5, -2.0, -2.75 }, result3);

    const first4 = [_]f64{ 1, -1, 2, 3, -1 };
    const second4 = [_]f64{ 1, 2, 3 };
    const result4 = try circularConvolution(allocator, &first4, &second4);
    defer allocator.free(result4);
    try testing.expectEqualSlices(f64, &[_]f64{ 8.0, -2.0, 3.0, 4.0, 11.0 }, result4);
}

test "circular convolution: empty and extreme cases" {
    const allocator = testing.allocator;

    const empty = [_]f64{};
    const empty_result = try circularConvolution(allocator, &empty, &empty);
    defer allocator.free(empty_result);
    try testing.expectEqual(@as(usize, 0), empty_result.len);

    const n: usize = 128;
    const ones_a = try allocator.alloc(f64, n);
    defer allocator.free(ones_a);
    const ones_b = try allocator.alloc(f64, n);
    defer allocator.free(ones_b);
    @memset(ones_a, 1.0);
    @memset(ones_b, 1.0);

    const extreme_result = try circularConvolution(allocator, ones_a, ones_b);
    defer allocator.free(extreme_result);
    for (extreme_result) |value| {
        try testing.expectApproxEqAbs(@as(f64, 128.0), value, 1e-9);
    }
}
