//! ELF Hash - Zig implementation
//! Reference: https://github.com/TheAlgorithms/Python/blob/master/hashes/elf.py

const std = @import("std");
const testing = std.testing;

/// Computes ELF hash.
///
/// Time complexity: O(n)
/// Space complexity: O(1)
pub fn elfHash(data: []const u8) u64 {
    var hash_value: u64 = 0;
    for (data) |letter| {
        hash_value = (hash_value << 4) +% letter;
        const x = hash_value & 0xF0000000;
        if (x != 0) {
            hash_value ^= x >> 24;
        }
        hash_value &= ~x;
    }
    return hash_value;
}

test "elf hash: python example and edge cases" {
    try testing.expectEqual(@as(u64, 253956621), elfHash("lorem ipsum"));
    try testing.expectEqual(@as(u64, 0), elfHash(""));

    const large = "0123456789abcdef" ** 5000;
    _ = elfHash(large);
}
