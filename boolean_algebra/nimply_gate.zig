//! NIMPLY Gate - Zig implementation
//! Reference: https://github.com/TheAlgorithms/Python/blob/master/boolean_algebra/nimply_gate.py

const std = @import("std");
const testing = std.testing;

/// Returns NIMPLY output: true only for (1, 0).
///
/// Time complexity: O(1)
/// Space complexity: O(1)
pub fn nimplyGate(input_1: i64, input_2: i64) u8 {
    return @intFromBool(input_1 != 0 and input_2 == 0);
}

test "nimply gate: truth table" {
    try testing.expectEqual(@as(u8, 0), nimplyGate(0, 0));
    try testing.expectEqual(@as(u8, 0), nimplyGate(0, 1));
    try testing.expectEqual(@as(u8, 1), nimplyGate(1, 0));
    try testing.expectEqual(@as(u8, 0), nimplyGate(1, 1));
}

test "nimply gate: non-binary values" {
    try testing.expectEqual(@as(u8, 1), nimplyGate(2, 0));
    try testing.expectEqual(@as(u8, 0), nimplyGate(1, -3));
    try testing.expectEqual(@as(u8, 1), nimplyGate(-1, 0));
}
