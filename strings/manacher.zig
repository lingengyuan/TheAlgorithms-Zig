//! Manacher - Zig implementation
//! Reference: https://github.com/TheAlgorithms/Python/blob/master/strings/manacher.py

const std = @import("std");
const testing = std.testing;

/// Returns longest palindromic substring in linear time.
/// Caller owns returned slice.
/// Time complexity: O(n), Space complexity: O(n)
pub fn palindromicString(allocator: std.mem.Allocator, inputString: []const u8) ![]u8 {
    if (inputString.len == 0) return try allocator.alloc(u8, 0);

    const transformed_len = inputString.len * 2 + 1;
    const transformed = try allocator.alloc(u8, transformed_len);
    defer allocator.free(transformed);

    for (inputString, 0..) |ch, i| {
        transformed[2 * i] = '|';
        transformed[2 * i + 1] = ch;
    }
    transformed[transformed_len - 1] = '|';

    const radius = try allocator.alloc(usize, transformed_len);
    defer allocator.free(radius);
    @memset(radius, 0);

    var center: usize = 0;
    var right: usize = 0;
    var best_center: usize = 0;
    var best_len: usize = 0;

    for (0..transformed_len) |i| {
        if (i < right and 2 * center >= i) {
            const mirror = 2 * center - i;
            radius[i] = @min(radius[mirror], right - i);
        }

        while (i + radius[i] + 1 < transformed_len and
            radius[i] + 1 <= i and
            transformed[i + radius[i] + 1] == transformed[i - (radius[i] + 1)])
        {
            radius[i] += 1;
        }

        if (i + radius[i] > right) {
            center = i;
            right = i + radius[i];
        }

        if (radius[i] > best_len) {
            best_len = radius[i];
            best_center = i;
        }
    }

    const start = (best_center - best_len) / 2;
    const out = try allocator.alloc(u8, best_len);
    @memcpy(out, inputString[start .. start + best_len]);
    return out;
}

test "manacher: python reference examples" {
    const alloc = testing.allocator;

    const s1 = try palindromicString(alloc, "abbbaba");
    defer alloc.free(s1);
    try testing.expectEqualStrings("abbba", s1);

    const s2 = try palindromicString(alloc, "ababa");
    defer alloc.free(s2);
    try testing.expectEqualStrings("ababa", s2);
}

test "manacher: boundary cases" {
    const alloc = testing.allocator;

    const empty = try palindromicString(alloc, "");
    defer alloc.free(empty);
    try testing.expectEqualStrings("", empty);

    const single = try palindromicString(alloc, "z");
    defer alloc.free(single);
    try testing.expectEqualStrings("z", single);

    const even_pal = try palindromicString(alloc, "cbbd");
    defer alloc.free(even_pal);
    try testing.expectEqualStrings("bb", even_pal);
}

test "manacher: extreme long input" {
    const alloc = testing.allocator;

    var long_text = [_]u8{'a'} ** 200_000;
    const out = try palindromicString(alloc, &long_text);
    defer alloc.free(out);

    try testing.expectEqual(@as(usize, 200_000), out.len);
    try testing.expect(out[0] == 'a');
    try testing.expect(out[out.len - 1] == 'a');
}
