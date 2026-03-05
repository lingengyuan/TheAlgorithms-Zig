//! 2-to-1 Multiplexer - Zig implementation
//! Reference: https://github.com/TheAlgorithms/Python/blob/master/boolean_algebra/multiplexer.py

const std = @import("std");
const testing = std.testing;

pub const MultiplexerError = error{InvalidInput};

fn isBit(value: i64) bool {
    return value == 0 or value == 1;
}

/// Implements a 2-to-1 multiplexer.
///
/// Time complexity: O(1)
/// Space complexity: O(1)
pub fn mux(input0: i64, input1: i64, select: i64) MultiplexerError!u8 {
    if (!isBit(input0) or !isBit(input1) or !isBit(select)) {
        return MultiplexerError.InvalidInput;
    }
    return if (select == 1) @intCast(input1) else @intCast(input0);
}

test "multiplexer: python examples" {
    try testing.expectEqual(@as(u8, 0), try mux(0, 1, 0));
    try testing.expectEqual(@as(u8, 1), try mux(0, 1, 1));
    try testing.expectEqual(@as(u8, 1), try mux(1, 0, 0));
    try testing.expectEqual(@as(u8, 0), try mux(1, 0, 1));
}

test "multiplexer: invalid values" {
    try testing.expectError(MultiplexerError.InvalidInput, mux(2, 1, 0));
    try testing.expectError(MultiplexerError.InvalidInput, mux(0, -1, 0));
    try testing.expectError(MultiplexerError.InvalidInput, mux(0, 1, 2));
}

test "multiplexer: extreme repeated checks" {
    var i: usize = 0;
    while (i < 10_000) : (i += 1) {
        try testing.expectEqual(@as(u8, 1), try mux(1, 1, @intCast(i % 2)));
    }
}
