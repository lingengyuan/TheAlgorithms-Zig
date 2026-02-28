//! Rabin-Karp String Search - Zig implementation
//! Reference: https://github.com/TheAlgorithms/Python/blob/master/strings/rabin_karp.py

const std = @import("std");
const testing = std.testing;

const BASE: u64 = 256;
const MOD: u64 = 1_000_003;

/// Returns true if pattern appears in text (Rabin-Karp rolling hash).
/// Time complexity: O(n + m) average, O(n·m) worst case
pub fn rabinKarp(text: []const u8, pattern: []const u8) bool {
    const n = text.len;
    const m = pattern.len;
    if (m > n) return false;
    if (m == 0) return true;

    // Precompute BASE^(m-1) mod MOD
    var base_pow: u64 = 1;
    for (0..m - 1) |_| base_pow = base_pow * BASE % MOD;

    var p_hash: u64 = 0;
    var t_hash: u64 = 0;
    for (0..m) |i| {
        p_hash = (p_hash * BASE + pattern[i]) % MOD;
        t_hash = (t_hash * BASE + text[i]) % MOD;
    }

    for (0..n - m + 1) |i| {
        if (t_hash == p_hash and std.mem.eql(u8, text[i..][0..m], pattern)) {
            return true;
        }
        if (i < n - m) {
            t_hash = (t_hash + MOD - text[i] * base_pow % MOD) % MOD;
            t_hash = (t_hash * BASE + text[i + m]) % MOD;
        }
    }
    return false;
}

test "rabin karp: found" {
    try testing.expect(rabinKarp("ABAAABCDBBABCDDEBCABC", "ABC"));
    try testing.expect(rabinKarp("hello world", "world"));
    try testing.expect(rabinKarp("TEST", "TEST"));
}

test "rabin karp: not found" {
    try testing.expect(!rabinKarp("ABC", "ABAAABCDBBABCDDEBCABC"));
    try testing.expect(!rabinKarp("hello", "world"));
    try testing.expect(!rabinKarp("", "ABC"));
}

test "rabin karp: empty pattern" {
    try testing.expect(rabinKarp("hello", ""));
}
