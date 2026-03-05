//! Find Previous Power of Two - Zig implementation
//! Reference: https://github.com/TheAlgorithms/Python/blob/master/bit_manipulation/find_previous_power_of_two.py

const std = @import("std");
const testing = std.testing;

pub const PreviousPowerError = error{NegativeInput};

/// Returns largest power of two less than or equal to `number`.
///
/// Time complexity: O(log n)
/// Space complexity: O(1)
pub fn findPreviousPowerOfTwo(number: i64) PreviousPowerError!u64 {
    if (number < 0) return PreviousPowerError.NegativeInput;
    if (number == 0) return 0;

    const value: u64 = @intCast(number);
    var power: u64 = 1;

    while (power <= value) {
        power <<= 1;
    }

    return if (value > 1) power >> 1 else 1;
}

test "find previous power of two: python sequence" {
    const expected = [_]u64{ 0, 1, 2, 2, 4, 4, 4, 4, 8, 8, 8, 8, 8, 8, 8, 8, 16, 16 };
    for (0..expected.len) |i| {
        try testing.expectEqual(expected[i], try findPreviousPowerOfTwo(@intCast(i)));
    }
}

test "find previous power of two: invalid input" {
    try testing.expectError(PreviousPowerError.NegativeInput, findPreviousPowerOfTwo(-5));
}

test "find previous power of two: extreme values" {
    try testing.expectEqual(@as(u64, 1), try findPreviousPowerOfTwo(1));
    try testing.expectEqual(@as(u64, 1) << 62, try findPreviousPowerOfTwo(std.math.maxInt(i64)));
}
