//! Reverse Words - Zig implementation
//! Reference: https://github.com/TheAlgorithms/Python/blob/master/strings/reverse_words.py

const std = @import("std");
const testing = std.testing;

/// Reverses the order of words in a sentence. Extra whitespace between words is collapsed.
/// Caller owns the returned slice.
pub fn reverseWords(allocator: std.mem.Allocator, sentence: []const u8) ![]u8 {
    // Split into words (skip whitespace)
    var words = std.ArrayListUnmanaged([]const u8){};
    defer words.deinit(allocator);

    var it = std.mem.splitScalar(u8, sentence, ' ');
    while (it.next()) |w| {
        if (w.len > 0) try words.append(allocator, w);
    }

    if (words.items.len == 0) {
        return try allocator.alloc(u8, 0);
    }

    // Calculate total length
    var total: usize = 0;
    for (words.items) |w| total += w.len;
    total += words.items.len - 1; // spaces between words

    const result = try allocator.alloc(u8, total);
    var pos: usize = 0;
    var i: usize = words.items.len;
    while (i > 0) {
        i -= 1;
        const w = words.items[i];
        @memcpy(result[pos..][0..w.len], w);
        pos += w.len;
        if (i > 0) {
            result[pos] = ' ';
            pos += 1;
        }
    }
    return result;
}

test "reverse words: basic" {
    const alloc = testing.allocator;
    const s = try reverseWords(alloc, "I love Python");
    defer alloc.free(s);
    try testing.expectEqualStrings("Python love I", s);
}

test "reverse words: extra spaces collapsed" {
    const alloc = testing.allocator;
    const s = try reverseWords(alloc, "I     Love          Python");
    defer alloc.free(s);
    try testing.expectEqualStrings("Python Love I", s);
}

test "reverse words: single word" {
    const alloc = testing.allocator;
    const s = try reverseWords(alloc, "hello");
    defer alloc.free(s);
    try testing.expectEqualStrings("hello", s);
}

test "reverse words: empty" {
    const alloc = testing.allocator;
    const s = try reverseWords(alloc, "");
    defer alloc.free(s);
    try testing.expectEqual(@as(usize, 0), s.len);
}
