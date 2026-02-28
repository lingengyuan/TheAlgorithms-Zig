//! Is Power of Two - Zig implementation
//! Reference: https://github.com/TheAlgorithms/Python/blob/master/bit_manipulation/is_power_of_two.py

const std = @import("std");
const testing = std.testing;

/// Returns true if n is a power of two (including 1 = 2^0).
/// Uses the bit trick: n & (n-1) == 0.
/// Time complexity: O(1)
pub fn isPowerOfTwo(n: u64) bool {
    if (n == 0) return false;
    return (n & (n - 1)) == 0;
}

test "is power of two: powers of two" {
    const powers = [_]u64{ 1, 2, 4, 8, 16, 32, 64, 128, 256, 1024, 1 << 32 };
    for (powers) |p| try testing.expect(isPowerOfTwo(p));
}

test "is power of two: non-powers" {
    const non_powers = [_]u64{ 3, 5, 6, 7, 9, 10, 100, 1000 };
    for (non_powers) |n| try testing.expect(!isPowerOfTwo(n));
}

test "is power of two: zero" {
    try testing.expect(!isPowerOfTwo(0));
}
