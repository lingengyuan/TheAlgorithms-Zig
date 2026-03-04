//! Uppercase ASCII - Zig implementation
//! Reference: https://github.com/TheAlgorithms/Python/blob/master/strings/upper.py

const std = @import("std");
const testing = std.testing;

/// Converts ASCII lowercase letters to uppercase.
/// Caller owns returned slice.
/// Time complexity: O(n), Space complexity: O(n)
pub fn upper(allocator: std.mem.Allocator, word: []const u8) ![]u8 {
    const out = try allocator.alloc(u8, word.len);
    for (word, 0..) |char, i| {
        out[i] = if (char >= 'a' and char <= 'z') char - 32 else char;
    }
    return out;
}

test "upper: python reference examples" {
    const alloc = testing.allocator;

    const s1 = try upper(alloc, "wow");
    defer alloc.free(s1);
    try testing.expectEqualStrings("WOW", s1);

    const s2 = try upper(alloc, "Hello");
    defer alloc.free(s2);
    try testing.expectEqualStrings("HELLO", s2);

    const s3 = try upper(alloc, "WHAT");
    defer alloc.free(s3);
    try testing.expectEqualStrings("WHAT", s3);

    const s4 = try upper(alloc, "wh[]32");
    defer alloc.free(s4);
    try testing.expectEqualStrings("WH[]32", s4);
}

test "upper: edge and extreme cases" {
    const empty = try upper(testing.allocator, "");
    defer testing.allocator.free(empty);
    try testing.expectEqual(@as(usize, 0), empty.len);

    var long_input = [_]u8{'z'} ** 200_000;
    const out = try upper(testing.allocator, &long_input);
    defer testing.allocator.free(out);
    try testing.expect(out[0] == 'Z');
    try testing.expect(out[out.len - 1] == 'Z');
}
