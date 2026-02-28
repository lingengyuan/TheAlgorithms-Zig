//! Integer Square Root (Newton's Method) - Zig implementation
//! Reference: https://github.com/TheAlgorithms/Python/blob/master/maths/integer_square_root.py

const std = @import("std");
const testing = std.testing;

/// Returns floor(sqrt(n)).
/// Time complexity: O(log n), Space complexity: O(1)
pub fn integerSquareRoot(n: u64) u64 {
    if (n < 2) return n;
    var x = n;
    var y: u64 = @intCast((@as(u128, x) + @as(u128, n) / x) / 2);
    while (y < x) {
        x = y;
        y = @intCast((@as(u128, x) + @as(u128, n) / x) / 2);
    }
    return x;
}

test "integer square root: perfect squares" {
    try testing.expectEqual(@as(u64, 0), integerSquareRoot(0));
    try testing.expectEqual(@as(u64, 1), integerSquareRoot(1));
    try testing.expectEqual(@as(u64, 4), integerSquareRoot(16));
    try testing.expectEqual(@as(u64, 9), integerSquareRoot(81));
}

test "integer square root: non-perfect squares" {
    try testing.expectEqual(@as(u64, 2), integerSquareRoot(8));
    try testing.expectEqual(@as(u64, 3), integerSquareRoot(15));
    try testing.expectEqual(@as(u64, 10), integerSquareRoot(108));
}

test "integer square root: large value" {
    try testing.expectEqual(@as(u64, 3037000499), integerSquareRoot(9_223_372_030_926_249_001));
}

test "integer square root: max u64" {
    try testing.expectEqual(@as(u64, 4_294_967_295), integerSquareRoot(std.math.maxInt(u64)));
}
