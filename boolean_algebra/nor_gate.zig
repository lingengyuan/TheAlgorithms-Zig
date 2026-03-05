//! NOR Gate - Zig implementation
//! Reference: https://github.com/TheAlgorithms/Python/blob/master/boolean_algebra/nor_gate.py

const std = @import("std");
const testing = std.testing;

/// Returns NOR result: 1 only when both inputs are zero.
///
/// Time complexity: O(1)
/// Space complexity: O(1)
pub fn norGate(input_1: anytype, input_2: anytype) u8 {
    return if (input_1 == 0 and input_2 == 0) 1 else 0;
}

test "nor gate: truth table" {
    try testing.expectEqual(@as(u8, 1), norGate(0, 0));
    try testing.expectEqual(@as(u8, 0), norGate(0, 1));
    try testing.expectEqual(@as(u8, 0), norGate(1, 0));
    try testing.expectEqual(@as(u8, 0), norGate(1, 1));
}

test "nor gate: python non-int examples" {
    try testing.expectEqual(@as(u8, 1), norGate(0.0, 0.0));
    try testing.expectEqual(@as(u8, 0), norGate(0, -7));
}
