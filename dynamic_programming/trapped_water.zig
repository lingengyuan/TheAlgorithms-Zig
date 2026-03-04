//! Trapped Rainwater (DP Prefix/Suffix Max) - Zig implementation
//! Reference: https://github.com/TheAlgorithms/Python/blob/master/dynamic_programming/trapped_water.py

const std = @import("std");
const testing = std.testing;
const Allocator = std.mem.Allocator;

pub const TrappedRainwaterError = error{
    NegativeHeight,
    Overflow,
};

/// Returns total trapped rainwater for elevation bars.
/// Heights must be non-negative.
/// Time complexity: O(n), Space complexity: O(n)
pub fn trappedRainwater(
    allocator: Allocator,
    heights: []const i64,
) (TrappedRainwaterError || Allocator.Error)!u64 {
    if (heights.len == 0) return 0;
    for (heights) |h| {
        if (h < 0) return TrappedRainwaterError.NegativeHeight;
    }

    const n = heights.len;
    const left_max = try allocator.alloc(i64, n);
    defer allocator.free(left_max);
    const right_max = try allocator.alloc(i64, n);
    defer allocator.free(right_max);

    left_max[0] = heights[0];
    for (1..n) |i| {
        left_max[i] = @max(heights[i], left_max[i - 1]);
    }

    right_max[n - 1] = heights[n - 1];
    var i = n - 1;
    while (i > 0) {
        i -= 1;
        right_max[i] = @max(heights[i], right_max[i + 1]);
    }

    var total: u64 = 0;
    for (0..n) |k| {
        const cap = @min(left_max[k], right_max[k]);
        const stored = cap - heights[k];
        const add: u64 = @intCast(stored);
        const next = @addWithOverflow(total, add);
        if (next[1] != 0) return TrappedRainwaterError.Overflow;
        total = next[0];
    }

    return total;
}

test "trapped rainwater: python examples" {
    try testing.expectEqual(@as(u64, 6), try trappedRainwater(testing.allocator, &[_]i64{ 0, 1, 0, 2, 1, 0, 1, 3, 2, 1, 2, 1 }));
    try testing.expectEqual(@as(u64, 9), try trappedRainwater(testing.allocator, &[_]i64{ 7, 1, 5, 3, 6, 4 }));
}

test "trapped rainwater: boundary and invalid input" {
    try testing.expectEqual(@as(u64, 0), try trappedRainwater(testing.allocator, &[_]i64{}));
    try testing.expectEqual(@as(u64, 0), try trappedRainwater(testing.allocator, &[_]i64{5}));
    try testing.expectError(TrappedRainwaterError.NegativeHeight, trappedRainwater(testing.allocator, &[_]i64{ 7, 1, 5, 3, 6, -1 }));
}

test "trapped rainwater: extreme wide basin" {
    var heights: [10002]i64 = [_]i64{0} ** 10002;
    heights[0] = 1_000_000;
    heights[heights.len - 1] = 1_000_000;

    try testing.expectEqual(@as(u64, 10_000_000_000), try trappedRainwater(testing.allocator, &heights));
}

test "trapped rainwater: overflow detection" {
    const heights = [_]i64{
        std.math.maxInt(i64),
        0,
        std.math.maxInt(i64),
        0,
        std.math.maxInt(i64),
        0,
        std.math.maxInt(i64),
    };
    try testing.expectError(TrappedRainwaterError.Overflow, trappedRainwater(testing.allocator, &heights));
}
