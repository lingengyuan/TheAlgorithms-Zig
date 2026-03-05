//! Simple Moving Average - Zig implementation
//! Reference: https://github.com/TheAlgorithms/Python/blob/master/financial/simple_moving_average.py

const std = @import("std");
const testing = std.testing;

pub const SimpleMovingAverageError = error{
    WindowSizeMustBePositive,
};

/// Computes simple moving average for each position.
/// Output length equals input length; entries before full window are `null`.
///
/// Time complexity: O(n * window_size)
/// Space complexity: O(n)
pub fn simpleMovingAverage(
    allocator: std.mem.Allocator,
    data: []const f64,
    window_size: usize,
) (std.mem.Allocator.Error || SimpleMovingAverageError)![]?f64 {
    if (window_size < 1) return SimpleMovingAverageError.WindowSizeMustBePositive;

    const sma = try allocator.alloc(?f64, data.len);
    for (0..data.len) |i| {
        if (i < window_size - 1) {
            sma[i] = null;
        } else {
            var sum: f64 = 0;
            for (i + 1 - window_size..i + 1) |j| {
                sum += data[j];
            }
            sma[i] = sum / @as(f64, @floatFromInt(window_size));
        }
    }
    return sma;
}

test "simple moving average: python examples" {
    const alloc = testing.allocator;

    const sma = try simpleMovingAverage(alloc, &[_]f64{ 10, 12, 15, 13, 14, 16, 18, 17, 19, 21 }, 3);
    defer alloc.free(sma);
    const expected = [_]?f64{ null, null, 12.333333333333334, 13.333333333333334, 14.0, 14.333333333333334, 16.0, 17.0, 18.0, 19.0 };
    try testing.expectEqual(expected.len, sma.len);
    for (expected, 0..) |exp, i| {
        if (exp) |v| {
            try testing.expect(sma[i] != null);
            try testing.expectApproxEqAbs(v, sma[i].?, 1e-12);
        } else {
            try testing.expect(sma[i] == null);
        }
    }

    const sma2 = try simpleMovingAverage(alloc, &[_]f64{ 10, 12, 15 }, 5);
    defer alloc.free(sma2);
    try testing.expectEqual(@as(usize, 3), sma2.len);
    try testing.expect(sma2[0] == null and sma2[1] == null and sma2[2] == null);

    try testing.expectError(SimpleMovingAverageError.WindowSizeMustBePositive, simpleMovingAverage(alloc, &[_]f64{ 1, 2, 3 }, 0));
}

test "simple moving average: boundary and extreme values" {
    const alloc = testing.allocator;

    const empty = try simpleMovingAverage(alloc, &[_]f64{}, 1);
    defer alloc.free(empty);
    try testing.expectEqual(@as(usize, 0), empty.len);

    const large = try alloc.alloc(f64, 2000);
    defer alloc.free(large);
    @memset(large, 1.0);

    const stress = try simpleMovingAverage(alloc, large, 500);
    defer alloc.free(stress);
    try testing.expectEqual(large.len, stress.len);
    try testing.expect(stress[499] != null);
    try testing.expectApproxEqAbs(@as(f64, 1.0), stress[1999].?, 1e-12);
}
