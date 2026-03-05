//! AND Gate - Zig implementation
//! Reference: https://github.com/TheAlgorithms/Python/blob/master/boolean_algebra/and_gate.py

const std = @import("std");
const testing = std.testing;

/// Two-input boolean AND over integer truthiness (zero is false, non-zero is true).
///
/// Time complexity: O(1)
/// Space complexity: O(1)
pub fn andGate(input_1: i64, input_2: i64) u8 {
    return if (input_1 != 0 and input_2 != 0) 1 else 0;
}

/// N-input boolean AND over integer truthiness.
/// Matches Python's all() behavior for empty input: returns 1.
///
/// Time complexity: O(n)
/// Space complexity: O(1)
pub fn nInputAndGate(inputs: []const i64) u8 {
    for (inputs) |value| {
        if (value == 0) return 0;
    }
    return 1;
}

test "and gate: truth table" {
    try testing.expectEqual(@as(u8, 0), andGate(0, 0));
    try testing.expectEqual(@as(u8, 0), andGate(0, 1));
    try testing.expectEqual(@as(u8, 0), andGate(1, 0));
    try testing.expectEqual(@as(u8, 1), andGate(1, 1));
}

test "and gate: non-binary truthiness" {
    try testing.expectEqual(@as(u8, 1), andGate(2, -5));
    try testing.expectEqual(@as(u8, 0), andGate(2, 0));
}

test "n input and gate: normal and edge cases" {
    try testing.expectEqual(@as(u8, 0), nInputAndGate(&[_]i64{ 1, 0, 1, 1, 0 }));
    try testing.expectEqual(@as(u8, 1), nInputAndGate(&[_]i64{ 1, 1, 1, 1, 1 }));
    try testing.expectEqual(@as(u8, 1), nInputAndGate(&[_]i64{}));
}

test "n input and gate: extreme integer values" {
    try testing.expectEqual(@as(u8, 1), nInputAndGate(&[_]i64{ std.math.minInt(i64), std.math.maxInt(i64) }));
    try testing.expectEqual(@as(u8, 0), nInputAndGate(&[_]i64{ std.math.minInt(i64), 0, std.math.maxInt(i64) }));
}
