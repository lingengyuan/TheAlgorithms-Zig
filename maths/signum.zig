//! Signum Function - Zig implementation
//! Reference: https://github.com/TheAlgorithms/Python/blob/master/maths/signum.py

const std = @import("std");
const testing = std.testing;

/// Applies signum function on `num` and returns -1, 0, or 1.
/// Time complexity: O(1), Space complexity: O(1)
pub fn signum(num: f64) i8 {
    if (num < 0.0) return -1;
    return if (num == 0.0) 0 else 1;
}

test "signum: python reference examples" {
    try testing.expectEqual(@as(i8, -1), signum(-10));
    try testing.expectEqual(@as(i8, 1), signum(10));
    try testing.expectEqual(@as(i8, 0), signum(0));
    try testing.expectEqual(@as(i8, -1), signum(-20.5));
    try testing.expectEqual(@as(i8, 1), signum(20.5));
    try testing.expectEqual(@as(i8, -1), signum(-1e-6));
    try testing.expectEqual(@as(i8, 1), signum(1e-6));
}

test "signum: edge and extreme cases" {
    try testing.expectEqual(@as(i8, 1), signum(@floatFromInt(std.math.maxInt(i64))));
    try testing.expectEqual(@as(i8, -1), signum(@floatFromInt(std.math.minInt(i64))));
    try testing.expectEqual(@as(i8, 0), signum(-0.0));
}
