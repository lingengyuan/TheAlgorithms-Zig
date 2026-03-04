//! Title Case - Zig implementation
//! Reference: https://github.com/TheAlgorithms/Python/blob/master/strings/title.py

const std = @import("std");
const testing = std.testing;

const whitespace = " \t\n\r\x0b\x0c";

/// Converts one word to title case (first character uppercase, rest lowercase).
/// Caller owns returned slice.
/// Time complexity: O(n), Space complexity: O(n)
pub fn toTitleCase(allocator: std.mem.Allocator, word: []const u8) ![]u8 {
    if (word.len == 0) return try allocator.alloc(u8, 0);

    const out = try allocator.alloc(u8, word.len);
    out[0] = std.ascii.toUpper(word[0]);
    for (word[1..], 1..) |ch, i| {
        out[i] = std.ascii.toLower(ch);
    }
    return out;
}

/// Converts sentence to title case using whitespace tokenization.
/// Output uses single spaces between words, matching Python's split/join behavior.
/// Caller owns returned slice.
/// Time complexity: O(n), Space complexity: O(n)
pub fn sentenceToTitleCase(allocator: std.mem.Allocator, inputStr: []const u8) ![]u8 {
    var out = std.ArrayListUnmanaged(u8){};
    errdefer out.deinit(allocator);

    var tokenizer = std.mem.tokenizeAny(u8, inputStr, whitespace);
    var first_word = true;

    while (tokenizer.next()) |word| {
        if (!first_word) try out.append(allocator, ' ');
        if (word.len > 0) {
            try out.append(allocator, std.ascii.toUpper(word[0]));
            for (word[1..]) |ch| {
                try out.append(allocator, std.ascii.toLower(ch));
            }
        }
        first_word = false;
    }

    return try out.toOwnedSlice(allocator);
}

test "title case: python reference examples" {
    const alloc = testing.allocator;

    const w1 = try toTitleCase(alloc, "Aakash");
    defer alloc.free(w1);
    try testing.expectEqualStrings("Aakash", w1);

    const w2 = try toTitleCase(alloc, "aakash");
    defer alloc.free(w2);
    try testing.expectEqualStrings("Aakash", w2);

    const w3 = try toTitleCase(alloc, "AAKASH");
    defer alloc.free(w3);
    try testing.expectEqualStrings("Aakash", w3);

    const w4 = try toTitleCase(alloc, "aAkAsH");
    defer alloc.free(w4);
    try testing.expectEqualStrings("Aakash", w4);

    const s1 = try sentenceToTitleCase(alloc, "aAkAsH gIrI");
    defer alloc.free(s1);
    try testing.expectEqualStrings("Aakash Giri", s1);
}

test "title case: boundary behavior" {
    const alloc = testing.allocator;

    const empty_word = try toTitleCase(alloc, "");
    defer alloc.free(empty_word);
    try testing.expectEqualStrings("", empty_word);

    const empty_sentence = try sentenceToTitleCase(alloc, "");
    defer alloc.free(empty_sentence);
    try testing.expectEqualStrings("", empty_sentence);

    const compact = try sentenceToTitleCase(alloc, "  hello   WORLD\tzig ");
    defer alloc.free(compact);
    try testing.expectEqualStrings("Hello World Zig", compact);
}

test "title case: extreme long word" {
    const alloc = testing.allocator;

    var long_word = [_]u8{'Z'} ** 200_000;
    const titled = try toTitleCase(alloc, &long_word);
    defer alloc.free(titled);

    try testing.expectEqual(@as(usize, 200_000), titled.len);
    try testing.expect(titled[0] == 'Z');
    try testing.expect(titled[1] == 'z');
    try testing.expect(titled[titled.len - 1] == 'z');
}
