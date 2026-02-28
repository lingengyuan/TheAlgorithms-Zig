//! Reverse Bits (32-bit) - Zig implementation
//! Reference: https://github.com/TheAlgorithms/Python/blob/master/bit_manipulation/reverse_bits.py

const std = @import("std");
const testing = std.testing;

/// Reverses the bits of a 32-bit unsigned integer.
/// Time complexity: O(32) = O(1)
pub fn reverseBits(n: u32) u32 {
    var result: u32 = 0;
    var x = n;
    for (0..32) |_| {
        result = (result << 1) | (x & 1);
        x >>= 1;
    }
    return result;
}

test "reverse bits: known values" {
    // 9 = 0b00000000_00000000_00000000_00001001
    // reversed = 0b10010000_00000000_00000000_00000000 = 2415919104
    try testing.expectEqual(@as(u32, 2415919104), reverseBits(9));
    // 43 = 0b00101011 → reversed 32-bit = 0b11010100_00000000_00000000_00000000 = 3556769792
    try testing.expectEqual(@as(u32, 3556769792), reverseBits(43));
}

test "reverse bits: zero and max" {
    try testing.expectEqual(@as(u32, 0), reverseBits(0));
    try testing.expectEqual(std.math.maxInt(u32), reverseBits(std.math.maxInt(u32)));
}

test "reverse bits: involutory" {
    // reversing twice yields original
    const vals = [_]u32{ 1, 7, 255, 65535, 0xDEADBEEF };
    for (vals) |v| {
        try testing.expectEqual(v, reverseBits(reverseBits(v)));
    }
}
