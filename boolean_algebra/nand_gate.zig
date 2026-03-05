//! NAND Gate - Zig implementation
//! Reference: https://github.com/TheAlgorithms/Python/blob/master/boolean_algebra/nand_gate.py

const std = @import("std");
const testing = std.testing;

/// Calculates NAND gate output for two inputs using integer truthiness.
///
/// Time complexity: O(1)
/// Space complexity: O(1)
pub fn nandGate(input_1: i64, input_2: i64) u8 {
    const both_true = (input_1 != 0 and input_2 != 0);
    return if (both_true) 0 else 1;
}

test "nand gate: truth table" {
    try testing.expectEqual(@as(u8, 1), nandGate(0, 0));
    try testing.expectEqual(@as(u8, 1), nandGate(0, 1));
    try testing.expectEqual(@as(u8, 1), nandGate(1, 0));
    try testing.expectEqual(@as(u8, 0), nandGate(1, 1));
}

test "nand gate: non-binary values" {
    try testing.expectEqual(@as(u8, 0), nandGate(2, -3));
    try testing.expectEqual(@as(u8, 1), nandGate(2, 0));
    try testing.expectEqual(@as(u8, 1), nandGate(std.math.minInt(i64), 0));
}
