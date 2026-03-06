//! Adler-32 - Zig implementation
//! Reference: https://github.com/TheAlgorithms/Python/blob/master/hashes/adler32.py

const std = @import("std");
const testing = std.testing;

const mod_adler: u32 = 65_521;

/// Computes Adler-32 checksum for input bytes.
///
/// Time complexity: O(n)
/// Space complexity: O(1)
pub fn adler32(plain_text: []const u8) u32 {
    var a: u32 = 1;
    var b: u32 = 0;

    for (plain_text) |ch| {
        a = (a + ch) % mod_adler;
        b = (b + a) % mod_adler;
    }

    return (b << 16) | a;
}

test "adler32: python examples and edge cases" {
    try testing.expectEqual(@as(u32, 363791387), adler32("Algorithms"));
    try testing.expectEqual(@as(u32, 708642122), adler32("go adler em all"));

    try testing.expectEqual(@as(u32, 1), adler32(""));

    const large = "a" ** 10_000;
    _ = adler32(large);
}
