//! Word Patterns - Zig implementation
//! Reference: https://github.com/TheAlgorithms/Python/blob/master/strings/word_patterns.py

const std = @import("std");
const testing = std.testing;
const Allocator = std.mem.Allocator;

/// Returns the numeric appearance pattern of characters in `word`.
/// Caller owns the returned slice.
/// Time complexity: O(n), Space complexity: O(k)
pub fn getWordPattern(allocator: Allocator, word: []const u8) ![]u8 {
    var letter_nums = std.AutoHashMap(u8, usize).init(allocator);
    defer letter_nums.deinit();

    var out = std.ArrayListUnmanaged(u8){};
    defer out.deinit(allocator);

    var next_num: usize = 0;
    for (word, 0..) |char, index| {
        const upper = std.ascii.toUpper(char);
        const entry = try letter_nums.getOrPut(upper);
        if (!entry.found_existing) {
            entry.value_ptr.* = next_num;
            next_num += 1;
        }
        if (index > 0) try out.append(allocator, '.');
        try out.writer(allocator).print("{d}", .{entry.value_ptr.*});
    }

    return out.toOwnedSlice(allocator);
}

test "word patterns: python samples" {
    const empty = try getWordPattern(testing.allocator, "");
    defer testing.allocator.free(empty);
    try testing.expectEqualStrings("", empty);

    const one = try getWordPattern(testing.allocator, " ");
    defer testing.allocator.free(one);
    try testing.expectEqualStrings("0", one);

    const pattern = try getWordPattern(testing.allocator, "pattern");
    defer testing.allocator.free(pattern);
    try testing.expectEqualStrings("0.1.2.2.3.4.5", pattern);
}

test "word patterns: longer samples" {
    const one = try getWordPattern(testing.allocator, "word pattern");
    defer testing.allocator.free(one);
    try testing.expectEqualStrings("0.1.2.3.4.5.6.7.7.8.2.9", one);

    const two = try getWordPattern(testing.allocator, "get word pattern");
    defer testing.allocator.free(two);
    try testing.expectEqualStrings("0.1.2.3.4.5.6.7.3.8.9.2.2.1.6.10", two);
}
