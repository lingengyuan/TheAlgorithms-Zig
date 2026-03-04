//! Bitap String Match - Zig implementation
//! Reference: https://github.com/TheAlgorithms/Python/blob/master/strings/bitap_string_match.py

const std = @import("std");
const testing = std.testing;

pub const BitapError = error{InvalidCharacter};

fn lowercaseIndex(ch: u8) BitapError!u5 {
    if (ch < 'a' or ch > 'z') return BitapError.InvalidCharacter;
    return @intCast(ch - 'a');
}

/// Returns index of first pattern occurrence in text, or null if not found.
/// Only lowercase a-z inputs are supported (same domain as Python reference).
/// For patterns longer than 63, falls back to linear substring search.
/// Time complexity: O(n * m) worst case for fallback, O(n) for bitap core
/// Space complexity: O(1)
pub fn bitapStringMatch(text: []const u8, pattern: []const u8) BitapError!?usize {
    if (pattern.len == 0) return 0;
    if (pattern.len > text.len) return null;

    for (pattern) |ch| _ = try lowercaseIndex(ch);
    for (text) |ch| _ = try lowercaseIndex(ch);

    if (pattern.len > 63) {
        return std.mem.indexOf(u8, text, pattern);
    }

    var pattern_mask = [_]u64{~@as(u64, 0)} ** 26;
    for (pattern, 0..) |ch, i| {
        const idx = try lowercaseIndex(ch);
        pattern_mask[idx] &= ~(@as(u64, 1) << @as(u6, @intCast(i)));
    }

    var state: u64 = ~@as(u64, 1);
    const match_bit = @as(u64, 1) << @as(u6, @intCast(pattern.len));

    for (text, 0..) |ch, i| {
        const idx = try lowercaseIndex(ch);
        state |= pattern_mask[idx];
        state <<= 1;

        if ((state & match_bit) == 0) {
            return i + 1 - pattern.len;
        }
    }

    return null;
}

test "bitap string match: python reference examples" {
    try testing.expectEqual(@as(?usize, 5), try bitapStringMatch("abdabababc", "ababc"));
    try testing.expectEqual(@as(?usize, 0), try bitapStringMatch("aaaaaaaaaaaaaaaaaa", "a"));
    try testing.expectEqual(@as(?usize, 0), try bitapStringMatch("zxywsijdfosdfnso", "zxywsijdfosdfnso"));
    try testing.expectEqual(@as(?usize, 0), try bitapStringMatch("abdabababc", ""));
    try testing.expectEqual(@as(?usize, 9), try bitapStringMatch("abdabababc", "c"));
    try testing.expectEqual(@as(?usize, null), try bitapStringMatch("abdabababc", "fofosdfo"));
    try testing.expectEqual(@as(?usize, null), try bitapStringMatch("abdab", "fofosdfo"));
}

test "bitap string match: invalid input and boundary cases" {
    try testing.expectError(BitapError.InvalidCharacter, bitapStringMatch("abC", "abc"));
    try testing.expectError(BitapError.InvalidCharacter, bitapStringMatch("abc", "a1"));
    try testing.expectEqual(@as(?usize, null), try bitapStringMatch("", "a"));
}

test "bitap string match: extreme and fallback cases" {
    const alloc = testing.allocator;

    const long_text = try alloc.alloc(u8, 200_001);
    defer alloc.free(long_text);
    @memset(long_text, 'a');
    long_text[long_text.len - 1] = 'b';
    try testing.expectEqual(@as(?usize, 199_999), try bitapStringMatch(long_text, "ab"));

    const long_pattern = try alloc.alloc(u8, 80);
    defer alloc.free(long_pattern);
    @memset(long_pattern, 'a');
    long_pattern[79] = 'b';
    try testing.expectEqual(@as(?usize, 199_921), try bitapStringMatch(long_text, long_pattern));
}
