//! NOT Gate - Zig implementation
//! Reference: https://github.com/TheAlgorithms/Python/blob/master/boolean_algebra/not_gate.py

const std = @import("std");
const testing = std.testing;

/// Returns logical NOT for single input.
/// Output is 1 only when input equals 0.
///
/// Time complexity: O(1)
/// Space complexity: O(1)
pub fn notGate(input_1: anytype) u8 {
    return if (input_1 == 0) 1 else 0;
}

test "not gate: python examples" {
    try testing.expectEqual(@as(u8, 1), notGate(0));
    try testing.expectEqual(@as(u8, 0), notGate(1));
}

test "not gate: edge values" {
    try testing.expectEqual(@as(u8, 1), notGate(0.0));
    try testing.expectEqual(@as(u8, 0), notGate(-7));
    try testing.expectEqual(@as(u8, 0), notGate(std.math.maxInt(i64)));
}
