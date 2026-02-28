//! Is Power of Four - Zig implementation
//! Reference: https://github.com/TheAlgorithms/Python/blob/master/bit_manipulation/power_of_4.py

const std = @import("std");
const testing = std.testing;

/// Returns true if n is a power of 4.
/// A power of 4 must be: positive, a power of 2, AND its single set bit
/// must be at an even bit position (0, 2, 4, ...).
/// Mask 0x55555555 has bits set at all even positions.
/// Time complexity: O(1)
pub fn isPowerOfFour(n: u64) bool {
    if (n == 0) return false;
    const is_pow2 = (n & (n - 1)) == 0;
    const even_bit_mask: u64 = 0x5555555555555555; // even positions set
    return is_pow2 and (n & even_bit_mask) != 0;
}

test "is power of four: powers" {
    const powers = [_]u64{ 1, 4, 16, 64, 256, 1024, 4096 };
    for (powers) |p| try testing.expect(isPowerOfFour(p));
}

test "is power of four: non-powers" {
    const non = [_]u64{ 0, 2, 3, 5, 6, 7, 8, 15, 32, 100 };
    for (non) |n| try testing.expect(!isPowerOfFour(n));
}
