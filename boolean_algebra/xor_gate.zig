//! XOR Gate - Zig implementation
//! Reference: https://github.com/TheAlgorithms/Python/blob/master/boolean_algebra/xor_gate.py

const std = @import("std");
const testing = std.testing;

/// Calculates XOR gate output for two integer inputs.
/// Python reference semantics: output depends on whether count of literal zero
/// inputs is odd.
///
/// Time complexity: O(1)
/// Space complexity: O(1)
pub fn xorGate(input_1: i64, input_2: i64) u8 {
    var zero_count: u8 = 0;
    if (input_1 == 0) zero_count += 1;
    if (input_2 == 0) zero_count += 1;
    return zero_count % 2;
}

test "xor gate: truth table" {
    try testing.expectEqual(@as(u8, 0), xorGate(0, 0));
    try testing.expectEqual(@as(u8, 1), xorGate(0, 1));
    try testing.expectEqual(@as(u8, 1), xorGate(1, 0));
    try testing.expectEqual(@as(u8, 0), xorGate(1, 1));
}

test "xor gate: non-binary values follow python semantics" {
    try testing.expectEqual(@as(u8, 1), xorGate(2, 0));
    try testing.expectEqual(@as(u8, 0), xorGate(2, 3));
    try testing.expectEqual(@as(u8, 1), xorGate(std.math.minInt(i64), 0));
}
