//! Atbash Cipher - Zig implementation
//! Reference: https://github.com/TheAlgorithms/Python/blob/master/ciphers/atbash.py

const std = @import("std");
const testing = std.testing;
const Allocator = std.mem.Allocator;

/// Applies Atbash substitution to ASCII letters.
/// Upper/lower case are preserved; non-letters are unchanged.
/// Time complexity: O(n), Space complexity: O(n)
pub fn atbash(allocator: Allocator, input: []const u8) ![]u8 {
    const out = try allocator.alloc(u8, input.len);
    errdefer allocator.free(out);

    for (input, 0..) |ch, i| {
        if (ch >= 'A' and ch <= 'Z') {
            out[i] = @intCast(@as(i16, 'Z') - @as(i16, ch) + @as(i16, 'A'));
        } else if (ch >= 'a' and ch <= 'z') {
            out[i] = @intCast(@as(i16, 'z') - @as(i16, ch) + @as(i16, 'a'));
        } else {
            out[i] = ch;
        }
    }
    return out;
}

test "atbash: python samples" {
    const alloc = testing.allocator;

    const a = try atbash(alloc, "ABCDEFG");
    defer alloc.free(a);
    try testing.expectEqualStrings("ZYXWVUT", a);

    const b = try atbash(alloc, "aW;;123BX");
    defer alloc.free(b);
    try testing.expectEqualStrings("zD;;123YC", b);
}

test "atbash: involution round-trip" {
    const alloc = testing.allocator;
    const input = "with Space & Symbols 2026";

    const once = try atbash(alloc, input);
    defer alloc.free(once);
    const twice = try atbash(alloc, once);
    defer alloc.free(twice);

    try testing.expectEqualStrings(input, twice);
}

test "atbash: empty input" {
    const alloc = testing.allocator;
    const out = try atbash(alloc, "");
    defer alloc.free(out);
    try testing.expectEqual(@as(usize, 0), out.len);
}

test "atbash: extreme long input" {
    const alloc = testing.allocator;
    const n: usize = 4096;
    const input = try alloc.alloc(u8, n);
    defer alloc.free(input);
    @memset(input, 'a');

    const out = try atbash(alloc, input);
    defer alloc.free(out);

    try testing.expectEqual(@as(usize, n), out.len);
    try testing.expect(out[0] == 'z' and out[n - 1] == 'z');
}
