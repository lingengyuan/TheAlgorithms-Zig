//! Join Strings - Zig implementation
//! Reference: https://github.com/TheAlgorithms/Python/blob/master/strings/join.py

const std = @import("std");
const testing = std.testing;

/// Joins list of strings with a separator.
/// Caller owns returned slice.
/// Time complexity: O(total_len), Space complexity: O(total_len)
pub fn join(allocator: std.mem.Allocator, separator: []const u8, separated: []const []const u8) ![]u8 {
    if (separated.len == 0) return try allocator.alloc(u8, 0);

    var total_len: usize = 0;
    for (separated) |part| total_len += part.len;
    total_len += (separated.len - 1) * separator.len;

    const out = try allocator.alloc(u8, total_len);
    var pos: usize = 0;
    for (separated, 0..) |part, idx| {
        @memcpy(out[pos..][0..part.len], part);
        pos += part.len;
        if (idx + 1 < separated.len) {
            @memcpy(out[pos..][0..separator.len], separator);
            pos += separator.len;
        }
    }
    return out;
}

test "join: python reference examples" {
    const alloc = testing.allocator;

    const j1 = try join(alloc, "", &[_][]const u8{ "a", "b", "c", "d" });
    defer alloc.free(j1);
    try testing.expectEqualStrings("abcd", j1);

    const j2 = try join(alloc, "#", &[_][]const u8{ "a", "b", "c", "d" });
    defer alloc.free(j2);
    try testing.expectEqualStrings("a#b#c#d", j2);

    const j3 = try join(alloc, " ", &[_][]const u8{ "You", "are", "amazing!" });
    defer alloc.free(j3);
    try testing.expectEqualStrings("You are amazing!", j3);

    const j4 = try join(alloc, ",", &[_][]const u8{ "", "", "" });
    defer alloc.free(j4);
    try testing.expectEqualStrings(",,", j4);

    const j5 = try join(alloc, "-", &[_][]const u8{ "apple", "banana", "cherry" });
    defer alloc.free(j5);
    try testing.expectEqualStrings("apple-banana-cherry", j5);
}

test "join: edge and extreme cases" {
    const empty = try join(testing.allocator, "#", &[_][]const u8{});
    defer testing.allocator.free(empty);
    try testing.expectEqual(@as(usize, 0), empty.len);

    const single = try join(testing.allocator, "#", &[_][]const u8{"a"});
    defer testing.allocator.free(single);
    try testing.expectEqualStrings("a", single);

    var parts: [10_000][]const u8 = undefined;
    for (&parts) |*p| p.* = "x";
    const huge = try join(testing.allocator, ",", &parts);
    defer testing.allocator.free(huge);
    try testing.expectEqual(@as(usize, 19_999), huge.len);
}
