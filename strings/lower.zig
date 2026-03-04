//! Lowercase ASCII - Zig implementation
//! Reference: https://github.com/TheAlgorithms/Python/blob/master/strings/lower.py

const std = @import("std");
const testing = std.testing;

/// Converts ASCII uppercase letters to lowercase.
/// Caller owns returned slice.
/// Time complexity: O(n), Space complexity: O(n)
pub fn lower(allocator: std.mem.Allocator, word: []const u8) ![]u8 {
    const out = try allocator.alloc(u8, word.len);
    for (word, 0..) |char, i| {
        out[i] = if (char >= 'A' and char <= 'Z') char + 32 else char;
    }
    return out;
}

test "lower: python reference examples" {
    const alloc = testing.allocator;

    const s1 = try lower(alloc, "wow");
    defer alloc.free(s1);
    try testing.expectEqualStrings("wow", s1);

    const s2 = try lower(alloc, "HellZo");
    defer alloc.free(s2);
    try testing.expectEqualStrings("hellzo", s2);

    const s3 = try lower(alloc, "WHAT");
    defer alloc.free(s3);
    try testing.expectEqualStrings("what", s3);

    const s4 = try lower(alloc, "wh[]32");
    defer alloc.free(s4);
    try testing.expectEqualStrings("wh[]32", s4);

    const s5 = try lower(alloc, "whAT");
    defer alloc.free(s5);
    try testing.expectEqualStrings("what", s5);
}

test "lower: edge and extreme cases" {
    const empty = try lower(testing.allocator, "");
    defer testing.allocator.free(empty);
    try testing.expectEqual(@as(usize, 0), empty.len);

    var long_input = [_]u8{'A'} ** 200_000;
    const out = try lower(testing.allocator, &long_input);
    defer testing.allocator.free(out);
    try testing.expect(out[0] == 'a');
    try testing.expect(out[out.len - 1] == 'a');
}
