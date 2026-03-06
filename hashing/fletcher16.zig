//! Fletcher-16 - Zig implementation
//! Reference: https://github.com/TheAlgorithms/Python/blob/master/hashes/fletcher16.py

const std = @import("std");
const testing = std.testing;

/// Computes Fletcher-16 checksum on input bytes.
///
/// Time complexity: O(n)
/// Space complexity: O(1)
pub fn fletcher16(text: []const u8) u16 {
    var sum1: u16 = 0;
    var sum2: u16 = 0;

    for (text) |character| {
        sum1 = (sum1 + character) % 255;
        sum2 = (sum1 + sum2) % 255;
    }

    return (sum2 << 8) | sum1;
}

test "fletcher16: python examples and edge cases" {
    try testing.expectEqual(@as(u16, 6752), fletcher16("hello world"));
    try testing.expectEqual(@as(u16, 28347), fletcher16("onethousandfourhundredthirtyfour"));
    try testing.expectEqual(@as(u16, 5655), fletcher16("The quick brown fox jumps over the lazy dog."));

    try testing.expectEqual(@as(u16, 0), fletcher16(""));

    const large = "z" ** 20_000;
    _ = fletcher16(large);
}
