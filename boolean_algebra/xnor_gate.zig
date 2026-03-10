//! XNOR Gate - Zig implementation
//! Reference: https://github.com/TheAlgorithms/Python/blob/master/boolean_algebra/xnor_gate.py

const std = @import("std");
const testing = std.testing;

/// Returns XNOR output: 1 when inputs have the same boolean truth value, else 0.
///
/// Time complexity: O(1)
/// Space complexity: O(1)
pub fn xnorGate(input_1: i64, input_2: i64) u8 {
    return if ((input_1 != 0) == (input_2 != 0)) 1 else 0;
}

test "xnor gate: truth table" {
    try testing.expectEqual(@as(u8, 1), xnorGate(0, 0));
    try testing.expectEqual(@as(u8, 0), xnorGate(0, 1));
    try testing.expectEqual(@as(u8, 0), xnorGate(1, 0));
    try testing.expectEqual(@as(u8, 1), xnorGate(1, 1));
}

test "xnor gate: non-binary values" {
    try testing.expectEqual(@as(u8, 1), xnorGate(5, 5));
    try testing.expectEqual(@as(u8, 1), xnorGate(-1, 1));
    try testing.expectEqual(@as(u8, 0), xnorGate(5, 0));
}
