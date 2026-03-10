//! IMPLY Gate - Zig implementation
//! Reference: https://github.com/TheAlgorithms/Python/blob/master/boolean_algebra/imply_gate.py

const std = @import("std");
const testing = std.testing;

pub const ImplyGateError = error{InvalidInput};

/// Returns IMPLY output: `a -> b` is false only for `a = 1` and `b = 0`.
///
/// Time complexity: O(1)
/// Space complexity: O(1)
pub fn implyGate(input_1: i64, input_2: i64) u8 {
    return @intFromBool(input_1 == 0 or input_2 != 0);
}

/// Recursively-equivalent left-fold implication:
/// `(((a -> b) -> c) -> d) ...`
///
/// Time complexity: O(n)
/// Space complexity: O(1)
pub fn recursiveImplyList(input_list: []const i64) ImplyGateError!u8 {
    if (input_list.len < 2) return ImplyGateError.InvalidInput;

    var result = implyGate(input_list[0], input_list[1]);
    for (input_list[2..]) |value| {
        result = implyGate(@as(i64, result), value);
    }
    return result;
}

test "imply gate: truth table" {
    try testing.expectEqual(@as(u8, 1), implyGate(0, 0));
    try testing.expectEqual(@as(u8, 1), implyGate(0, 1));
    try testing.expectEqual(@as(u8, 0), implyGate(1, 0));
    try testing.expectEqual(@as(u8, 1), implyGate(1, 1));
}

test "imply gate: recursive python examples" {
    try testing.expectEqual(@as(u8, 1), try recursiveImplyList(&[_]i64{ 0, 0 }));
    try testing.expectEqual(@as(u8, 1), try recursiveImplyList(&[_]i64{ 0, 1 }));
    try testing.expectEqual(@as(u8, 0), try recursiveImplyList(&[_]i64{ 1, 0 }));
    try testing.expectEqual(@as(u8, 1), try recursiveImplyList(&[_]i64{ 1, 1 }));

    try testing.expectEqual(@as(u8, 0), try recursiveImplyList(&[_]i64{ 0, 0, 0 }));
    try testing.expectEqual(@as(u8, 1), try recursiveImplyList(&[_]i64{ 0, 0, 1 }));
    try testing.expectEqual(@as(u8, 0), try recursiveImplyList(&[_]i64{ 0, 1, 0 }));
    try testing.expectEqual(@as(u8, 1), try recursiveImplyList(&[_]i64{ 0, 1, 1 }));
    try testing.expectEqual(@as(u8, 1), try recursiveImplyList(&[_]i64{ 1, 0, 0 }));
    try testing.expectEqual(@as(u8, 1), try recursiveImplyList(&[_]i64{ 1, 0, 1 }));
    try testing.expectEqual(@as(u8, 0), try recursiveImplyList(&[_]i64{ 1, 1, 0 }));
    try testing.expectEqual(@as(u8, 1), try recursiveImplyList(&[_]i64{ 1, 1, 1 }));
}

test "imply gate: invalid and non-binary edge values" {
    try testing.expectError(ImplyGateError.InvalidInput, recursiveImplyList(&[_]i64{}));
    try testing.expectError(ImplyGateError.InvalidInput, recursiveImplyList(&[_]i64{1}));

    try testing.expectEqual(@as(u8, 1), implyGate(0, 9));
    try testing.expectEqual(@as(u8, 1), implyGate(-3, 1));
    try testing.expectEqual(@as(u8, 0), implyGate(2, 0));
    try testing.expectEqual(@as(u8, 1), implyGate(1, 5));
}
