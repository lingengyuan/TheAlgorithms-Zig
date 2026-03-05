//! OR Gate - Zig implementation
//! Reference: https://github.com/TheAlgorithms/Python/blob/master/boolean_algebra/or_gate.py

const std = @import("std");
const testing = std.testing;

/// Calculates OR gate output for two integer inputs.
/// Python reference semantics: only literal `1` counts as logical true.
///
/// Time complexity: O(1)
/// Space complexity: O(1)
pub fn orGate(input_1: i64, input_2: i64) u8 {
    return if (input_1 == 1 or input_2 == 1) 1 else 0;
}

test "or gate: truth table" {
    try testing.expectEqual(@as(u8, 0), orGate(0, 0));
    try testing.expectEqual(@as(u8, 1), orGate(0, 1));
    try testing.expectEqual(@as(u8, 1), orGate(1, 0));
    try testing.expectEqual(@as(u8, 1), orGate(1, 1));
}

test "or gate: non-binary values follow python semantics" {
    try testing.expectEqual(@as(u8, 0), orGate(2, 0));
    try testing.expectEqual(@as(u8, 1), orGate(2, 1));
    try testing.expectEqual(@as(u8, 0), orGate(std.math.maxInt(i64), std.math.minInt(i64)));
}
