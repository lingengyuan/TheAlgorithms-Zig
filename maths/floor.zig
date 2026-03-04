//! Floor Function - Zig implementation
//! Reference: https://github.com/TheAlgorithms/Python/blob/master/maths/floor.py

const std = @import("std");
const testing = std.testing;

/// Returns floor of `x`.
/// Time complexity: O(1), Space complexity: O(1)
pub fn floor(x: f64) i128 {
    const truncated: i128 = @intFromFloat(@trunc(x));
    const truncated_f: f64 = @floatFromInt(truncated);
    return if (x - truncated_f >= 0.0) truncated else truncated - 1;
}

test "floor: python reference examples" {
    const inputs = [_]f64{ 1, -1, 0, -0.0, 1.1, -1.1, 1.0, -1.0, 1_000_000_000 };
    for (inputs) |x| {
        try testing.expectEqual(@as(i128, @intFromFloat(@floor(x))), floor(x));
    }
}

test "floor: edge and extreme cases" {
    try testing.expectEqual(@as(i128, -2), floor(-1.0000000001));
    try testing.expectEqual(@as(i128, 9_007_199_254_740_991), floor(9_007_199_254_740_991.0)); // 2^53-1
    try testing.expectEqual(@as(i128, -9_007_199_254_740_991), floor(-9_007_199_254_740_991.0));
}
