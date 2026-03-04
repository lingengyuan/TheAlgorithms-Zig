//! Capitalize - Zig implementation
//! Reference: https://github.com/TheAlgorithms/Python/blob/master/strings/capitalize.py

const std = @import("std");
const testing = std.testing;

/// Capitalizes the first character of a sentence.
/// Caller owns returned slice.
/// Time complexity: O(n), Space complexity: O(n)
pub fn capitalize(allocator: std.mem.Allocator, sentence: []const u8) ![]u8 {
    if (sentence.len == 0) return try allocator.alloc(u8, 0);

    const out = try allocator.alloc(u8, sentence.len);
    @memcpy(out, sentence);
    out[0] = std.ascii.toUpper(out[0]);
    return out;
}

test "capitalize: python reference examples" {
    const alloc = testing.allocator;

    const s1 = try capitalize(alloc, "hello world");
    defer alloc.free(s1);
    try testing.expectEqualStrings("Hello world", s1);

    const s2 = try capitalize(alloc, "123 hello world");
    defer alloc.free(s2);
    try testing.expectEqualStrings("123 hello world", s2);

    const s3 = try capitalize(alloc, " hello world");
    defer alloc.free(s3);
    try testing.expectEqualStrings(" hello world", s3);

    const s4 = try capitalize(alloc, "a");
    defer alloc.free(s4);
    try testing.expectEqualStrings("A", s4);

    const s5 = try capitalize(alloc, "");
    defer alloc.free(s5);
    try testing.expectEqualStrings("", s5);
}

test "capitalize: edge and extreme cases" {
    var long_input = [_]u8{'a'} ** 100_000;
    const out = try capitalize(testing.allocator, &long_input);
    defer testing.allocator.free(out);
    try testing.expect(out[0] == 'A');
    try testing.expect(out[out.len - 1] == 'a');
}
