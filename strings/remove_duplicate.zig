//! Remove Duplicate Words - Zig implementation
//! Reference: https://github.com/TheAlgorithms/Python/blob/master/strings/remove_duplicate.py

const std = @import("std");
const testing = std.testing;

const whitespace = " \t\n\r\x0b\x0c";

fn lessThan(_: void, a: []const u8, b: []const u8) bool {
    return std.mem.order(u8, a, b) == .lt;
}

/// Splits by whitespace, deduplicates words, sorts, and rejoins with one space.
/// Caller owns returned slice.
/// Time complexity: O(k log k + n), Space complexity: O(k)
/// where k is number of unique words and n is sentence length.
pub fn removeDuplicates(allocator: std.mem.Allocator, sentence: []const u8) ![]u8 {
    var unique = std.StringHashMap(void).init(allocator);
    defer unique.deinit();

    var tokenizer = std.mem.tokenizeAny(u8, sentence, whitespace);
    while (tokenizer.next()) |word| {
        _ = try unique.getOrPutValue(word, {});
    }

    if (unique.count() == 0) {
        return try allocator.alloc(u8, 0);
    }

    const words = try allocator.alloc([]const u8, unique.count());
    defer allocator.free(words);

    var index: usize = 0;
    var iterator = unique.keyIterator();
    while (iterator.next()) |word_ptr| {
        words[index] = word_ptr.*;
        index += 1;
    }

    std.mem.sort([]const u8, words, {}, lessThan);

    var total_len: usize = words.len - 1;
    for (words) |word| total_len += word.len;

    const out = try allocator.alloc(u8, total_len);
    var write_index: usize = 0;
    for (words, 0..) |word, i| {
        @memcpy(out[write_index..][0..word.len], word);
        write_index += word.len;
        if (i + 1 < words.len) {
            out[write_index] = ' ';
            write_index += 1;
        }
    }
    return out;
}

test "remove duplicate: python reference examples" {
    const alloc = testing.allocator;

    const r1 = try removeDuplicates(alloc, "Python is great and Java is also great");
    defer alloc.free(r1);
    try testing.expectEqualStrings("Java Python also and great is", r1);

    const r2 = try removeDuplicates(alloc, "Python   is      great and Java is also great");
    defer alloc.free(r2);
    try testing.expectEqualStrings("Java Python also and great is", r2);
}

test "remove duplicate: boundary cases" {
    const alloc = testing.allocator;

    const empty = try removeDuplicates(alloc, "");
    defer alloc.free(empty);
    try testing.expectEqualStrings("", empty);

    const case_sensitive = try removeDuplicates(alloc, "a A a");
    defer alloc.free(case_sensitive);
    try testing.expectEqualStrings("A a", case_sensitive);
}

test "remove duplicate: extreme repeated input" {
    const alloc = testing.allocator;
    var builder = std.ArrayListUnmanaged(u8){};
    defer builder.deinit(alloc);

    for (0..30_000) |_| {
        try builder.appendSlice(alloc, "zig ");
    }

    const sentence = builder.items;
    const result = try removeDuplicates(alloc, sentence);
    defer alloc.free(result);
    try testing.expectEqualStrings("zig", result);
}
