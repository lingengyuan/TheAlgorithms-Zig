//! DJB2 Hash - Zig implementation
//! Reference: https://github.com/TheAlgorithms/Python/blob/master/hashes/djb2.py

const std = @import("std");
const testing = std.testing;

/// Computes djb2 hash (32-bit masked variant).
///
/// Time complexity: O(n)
/// Space complexity: O(1)
pub fn djb2(s: []const u8) u32 {
    var hash_value: u32 = 5381;
    for (s) |x| {
        hash_value = ((hash_value << 5) +% hash_value) +% x;
    }
    return hash_value;
}

test "djb2: python examples and edge cases" {
    try testing.expectEqual(@as(u32, 3782405311), djb2("Algorithms"));
    try testing.expectEqual(@as(u32, 1609059040), djb2("scramble bits"));

    try testing.expectEqual(@as(u32, 5381), djb2(""));

    const long = "x" ** 50_000;
    _ = djb2(long);
}
