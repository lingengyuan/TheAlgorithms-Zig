//! Strip - Zig implementation
//! Reference: https://github.com/TheAlgorithms/Python/blob/master/strings/strip.py

const std = @import("std");
const testing = std.testing;

/// Removes leading/trailing characters from string.
/// Time complexity: O(n), Space complexity: O(1)
pub fn strip(userString: []const u8, characters: []const u8) []const u8 {
    var start: usize = 0;
    var end: usize = userString.len;

    while (start < end and std.mem.indexOfScalar(u8, characters, userString[start]) != null) {
        start += 1;
    }
    while (end > start and std.mem.indexOfScalar(u8, characters, userString[end - 1]) != null) {
        end -= 1;
    }
    return userString[start..end];
}

/// Default Python behavior with whitespace character set.
pub fn stripWhitespace(userString: []const u8) []const u8 {
    return strip(userString, " \t\n\r");
}

test "strip: python reference examples" {
    try testing.expectEqualStrings("hello", stripWhitespace("   hello   "));
    try testing.expectEqualStrings("world", strip("...world...", "."));
    try testing.expectEqualStrings("hello", strip("123hello123", "123"));
    try testing.expectEqualStrings("", stripWhitespace(""));
}

test "strip: boundary and edge cases" {
    try testing.expectEqualStrings("data", strip("data", ""));
    try testing.expectEqualStrings("", strip("aaaa", "a"));
    try testing.expectEqualStrings("middle", strip("***middle***", "*"));
}

test "strip: extreme long input" {
    const alloc = testing.allocator;

    const prefix_len = 100_000;
    const middle_len = 200_000;
    const suffix_len = 100_000;
    const total_len = prefix_len + middle_len + suffix_len;

    const long = try alloc.alloc(u8, total_len);
    defer alloc.free(long);

    @memset(long[0..prefix_len], 'x');
    @memset(long[prefix_len .. prefix_len + middle_len], 'a');
    @memset(long[prefix_len + middle_len ..], 'x');

    const trimmed = strip(long, "x");
    try testing.expectEqual(@as(usize, middle_len), trimmed.len);
    try testing.expect(trimmed[0] == 'a');
    try testing.expect(trimmed[trimmed.len - 1] == 'a');
}
