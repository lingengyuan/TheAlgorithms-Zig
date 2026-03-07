//! SDBM Hash - Zig implementation
//! Reference: https://github.com/TheAlgorithms/Python/blob/master/hashes/sdbm.py

const std = @import("std");
const testing = std.testing;
const BigInt = std.math.big.int.Managed;

/// Computes SDBM hash using arbitrary-precision integer arithmetic
/// to match Python's unbounded integer behavior.
/// Caller owns returned decimal-string buffer.
///
/// Time complexity: O(n * k)
/// Space complexity: O(k)
/// where k is the bigint limb/decimal size of the running hash.
pub fn sdbm(allocator: std.mem.Allocator, plain_text: []const u8) ![]u8 {
    var hash_value = try BigInt.initSet(allocator, 0);
    defer hash_value.deinit();

    var shift_6 = try BigInt.initSet(allocator, 0);
    defer shift_6.deinit();

    var shift_16 = try BigInt.initSet(allocator, 0);
    defer shift_16.deinit();

    var next_hash = try BigInt.initSet(allocator, 0);
    defer next_hash.deinit();

    for (plain_text) |plain_chr| {
        try shift_6.shiftLeft(&hash_value, 6);
        try shift_16.shiftLeft(&hash_value, 16);

        try next_hash.add(&shift_6, &shift_16);
        try next_hash.sub(&next_hash, &hash_value);
        try next_hash.addScalar(&next_hash, plain_chr);

        std.mem.swap(BigInt, &hash_value, &next_hash);
    }

    return hash_value.toString(allocator, 10, .lower);
}

test "sdbm hash: python doctest values" {
    const out1 = try sdbm(testing.allocator, "Algorithms");
    defer testing.allocator.free(out1);
    try testing.expectEqualStrings("1462174910723540325254304520539387479031000036", out1);

    const out2 = try sdbm(testing.allocator, "scramble bits");
    defer testing.allocator.free(out2);
    try testing.expectEqualStrings("730247649148944819640658295400555317318720608290373040936089", out2);
}

test "sdbm hash: empty input and extreme deterministic case" {
    const empty_hash = try sdbm(testing.allocator, "");
    defer testing.allocator.free(empty_hash);
    try testing.expectEqualStrings("0", empty_hash);

    const repeated = "a" ** 4096;
    const hash_a = try sdbm(testing.allocator, repeated);
    defer testing.allocator.free(hash_a);

    const hash_b = try sdbm(testing.allocator, repeated);
    defer testing.allocator.free(hash_b);

    try testing.expectEqualStrings(hash_a, hash_b);
    try testing.expect(hash_a.len > 1000);
    for (hash_a) |ch| {
        try testing.expect(ch >= '0' and ch <= '9');
    }
}
