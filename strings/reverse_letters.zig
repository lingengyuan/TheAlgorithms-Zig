//! Reverse Letters - Zig implementation
//! Reference: https://github.com/TheAlgorithms/Python/blob/master/strings/reverse_letters.py

const std = @import("std");
const testing = std.testing;

const whitespace = " \t\n\r\x0b\x0c";

/// Reverses each word whose length is greater than `lengthThreshold`.
/// Words are split by whitespace and rejoined with single spaces.
/// Caller owns returned slice.
/// Time complexity: O(n), Space complexity: O(n)
pub fn reverseLetters(
    allocator: std.mem.Allocator,
    sentence: []const u8,
    lengthThreshold: usize,
) ![]u8 {
    var out = std.ArrayListUnmanaged(u8){};
    errdefer out.deinit(allocator);

    var tokenizer = std.mem.tokenizeAny(u8, sentence, whitespace);
    var first_word = true;

    while (tokenizer.next()) |word| {
        if (!first_word) try out.append(allocator, ' ');

        if (word.len > lengthThreshold) {
            var i = word.len;
            while (i > 0) {
                i -= 1;
                try out.append(allocator, word[i]);
            }
        } else {
            try out.appendSlice(allocator, word);
        }

        first_word = false;
    }

    return try out.toOwnedSlice(allocator);
}

/// Default behavior from Python reference, reverse all words longer than 0.
pub fn reverseLettersDefault(allocator: std.mem.Allocator, sentence: []const u8) ![]u8 {
    return reverseLetters(allocator, sentence, 0);
}

test "reverse letters: python reference examples" {
    const alloc = testing.allocator;

    const s1 = try reverseLetters(alloc, "Hey wollef sroirraw", 3);
    defer alloc.free(s1);
    try testing.expectEqualStrings("Hey fellow warriors", s1);

    const s2 = try reverseLetters(alloc, "nohtyP is nohtyP", 2);
    defer alloc.free(s2);
    try testing.expectEqualStrings("Python is Python", s2);

    const s3 = try reverseLettersDefault(alloc, "racecar");
    defer alloc.free(s3);
    try testing.expectEqualStrings("racecar", s3);

    const s4 = try reverseLetters(alloc, "1 12 123 1234 54321 654321", 0);
    defer alloc.free(s4);
    try testing.expectEqualStrings("1 21 321 4321 12345 123456", s4);
}

test "reverse letters: boundary cases" {
    const alloc = testing.allocator;

    const empty = try reverseLettersDefault(alloc, "");
    defer alloc.free(empty);
    try testing.expectEqualStrings("", empty);

    const compact = try reverseLetters(alloc, "  one   two  ", 10);
    defer alloc.free(compact);
    try testing.expectEqualStrings("one two", compact);
}

test "reverse letters: extreme long word" {
    const alloc = testing.allocator;
    var long_word = [_]u8{'x'} ** 200_000;
    long_word[0] = 'a';
    long_word[long_word.len - 1] = 'z';

    const out = try reverseLettersDefault(alloc, &long_word);
    defer alloc.free(out);

    try testing.expectEqual(@as(usize, 200_000), out.len);
    try testing.expect(out[0] == 'z');
    try testing.expect(out[out.len - 1] == 'a');
}
