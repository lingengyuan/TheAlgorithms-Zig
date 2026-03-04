//! Mixed Keyword Cipher - Zig implementation
//! Reference: https://github.com/TheAlgorithms/Python/blob/master/ciphers/mixed_keyword_cypher.py

const std = @import("std");
const testing = std.testing;
const Allocator = std.mem.Allocator;

pub const MixedKeywordError = error{EmptyKeyword};

const ALPHABET = "ABCDEFGHIJKLMNOPQRSTUVWXYZ";

/// Builds alphabet mapping used by mixed-keyword cipher.
/// Time complexity: O(26), Space complexity: O(26)
pub fn buildMapping(keyword: []const u8) ![26]u8 {
    var seen = [_]bool{false} ** 26;
    var unique = [_]u8{0} ** 26;
    var unique_len: usize = 0;

    for (keyword) |raw_ch| {
        const ch = std.ascii.toUpper(raw_ch);
        if (ch < 'A' or ch > 'Z') continue;
        const idx = ch - 'A';
        if (!seen[idx]) {
            seen[idx] = true;
            unique[unique_len] = ch;
            unique_len += 1;
        }
    }

    if (unique_len == 0) return MixedKeywordError.EmptyKeyword;

    var shifted = [_]u8{0} ** 26;
    var shifted_len: usize = 0;

    for (unique[0..unique_len]) |ch| {
        shifted[shifted_len] = ch;
        shifted_len += 1;
    }

    for (ALPHABET) |ch| {
        const idx = ch - 'A';
        if (!seen[idx]) {
            shifted[shifted_len] = ch;
            shifted_len += 1;
        }
    }

    var mapping = [_]u8{0} ** 26;
    var letter_index: usize = 0;

    for (0..unique_len) |column| {
        var row_start: usize = 0;
        while (row_start < 26) : (row_start += unique_len) {
            const row_len = @min(unique_len, 26 - row_start);
            if (row_len <= column) break;

            mapping[letter_index] = shifted[row_start + column];
            letter_index += 1;
        }
    }

    return mapping;
}

/// Encrypts plaintext with mixed-keyword substitution.
/// Output follows Python reference behavior (uppercase output).
/// Time complexity: O(n), Space complexity: O(n)
pub fn mixedKeywordEncrypt(allocator: Allocator, keyword: []const u8, plaintext: []const u8) ![]u8 {
    const mapping = try buildMapping(keyword);

    const out = try allocator.alloc(u8, plaintext.len);
    errdefer allocator.free(out);

    for (plaintext, 0..) |raw_ch, i| {
        const ch = std.ascii.toUpper(raw_ch);
        if (ch >= 'A' and ch <= 'Z') {
            out[i] = mapping[ch - 'A'];
        } else {
            out[i] = ch;
        }
    }

    return out;
}

test "mixed keyword: python sample" {
    const alloc = testing.allocator;

    const out = try mixedKeywordEncrypt(alloc, "college", "UNIVERSITY");
    defer alloc.free(out);
    try testing.expectEqualStrings("XKJGUFMJST", out);
}

test "mixed keyword: mapping basics" {
    const m = try buildMapping("HELLO");
    try testing.expectEqual(@as(u8, 'H'), m[0]);
    try testing.expectEqual(@as(u8, 'A'), m[1]);
    try testing.expectEqual(@as(u8, 'F'), m[2]);
}

test "mixed keyword: invalid and extreme" {
    const alloc = testing.allocator;
    try testing.expectError(MixedKeywordError.EmptyKeyword, mixedKeywordEncrypt(alloc, "123?!", "ABC"));

    const n: usize = 10000;
    const plain = try alloc.alloc(u8, n);
    defer alloc.free(plain);

    for (plain, 0..) |*ch, i| {
        ch.* = switch (i % 5) {
            0 => 'u',
            1 => 'N',
            2 => '!',
            3 => ' ',
            else => '3',
        };
    }

    const out = try mixedKeywordEncrypt(alloc, "ALGORITHM", plain);
    defer alloc.free(out);
    try testing.expectEqual(@as(usize, n), out.len);
}
