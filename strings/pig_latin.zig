//! Pig Latin - Zig implementation
//! Reference: https://github.com/TheAlgorithms/Python/blob/master/strings/pig_latin.py

const std = @import("std");
const testing = std.testing;

fn isVowel(ch: u8) bool {
    return ch == 'a' or ch == 'e' or ch == 'i' or ch == 'o' or ch == 'u';
}

fn isWhitespace(ch: u8) bool {
    return ch == ' ' or ch == '\t' or ch == '\n' or ch == '\r' or ch == '\x0b' or ch == '\x0c';
}

/// Converts a word to Pig Latin.
/// Returns empty string for null/blank input.
/// Caller owns returned slice.
/// Time complexity: O(n), Space complexity: O(n)
pub fn pigLatin(allocator: std.mem.Allocator, wordOpt: ?[]const u8) ![]u8 {
    const word = wordOpt orelse return try allocator.alloc(u8, 0);
    if (word.len == 0) return try allocator.alloc(u8, 0);

    var has_non_whitespace = false;
    for (word) |ch| {
        if (!isWhitespace(ch)) {
            has_non_whitespace = true;
            break;
        }
    }
    if (!has_non_whitespace) return try allocator.alloc(u8, 0);

    const lower = try allocator.alloc(u8, word.len);
    defer allocator.free(lower);
    for (word, 0..) |ch, i| lower[i] = std.ascii.toLower(ch);

    if (isVowel(lower[0])) {
        const out = try allocator.alloc(u8, lower.len + 3);
        @memcpy(out[0..lower.len], lower);
        @memcpy(out[lower.len..], "way");
        return out;
    }

    var first_vowel: ?usize = null;
    for (lower, 0..) |ch, i| {
        if (isVowel(ch)) {
            first_vowel = i;
            break;
        }
    }

    // Matches Python behavior when no vowel exists: it uses the loop's final index.
    const split = first_vowel orelse (lower.len - 1);
    const out = try allocator.alloc(u8, lower.len + 2);
    var write: usize = 0;

    @memcpy(out[write..][0 .. lower.len - split], lower[split..]);
    write += lower.len - split;
    @memcpy(out[write..][0..split], lower[0..split]);
    write += split;
    @memcpy(out[write..][0..2], "ay");

    return out;
}

test "pig latin: python reference examples" {
    const alloc = testing.allocator;

    const p1 = try pigLatin(alloc, "pig");
    defer alloc.free(p1);
    try testing.expectEqualStrings("igpay", p1);

    const p2 = try pigLatin(alloc, "latin");
    defer alloc.free(p2);
    try testing.expectEqualStrings("atinlay", p2);

    const p3 = try pigLatin(alloc, "banana");
    defer alloc.free(p3);
    try testing.expectEqualStrings("ananabay", p3);

    const p4 = try pigLatin(alloc, "friends");
    defer alloc.free(p4);
    try testing.expectEqualStrings("iendsfray", p4);

    const p5 = try pigLatin(alloc, "smile");
    defer alloc.free(p5);
    try testing.expectEqualStrings("ilesmay", p5);

    const p6 = try pigLatin(alloc, "string");
    defer alloc.free(p6);
    try testing.expectEqualStrings("ingstray", p6);

    const p7 = try pigLatin(alloc, "eat");
    defer alloc.free(p7);
    try testing.expectEqualStrings("eatway", p7);

    const p8 = try pigLatin(alloc, "omelet");
    defer alloc.free(p8);
    try testing.expectEqualStrings("omeletway", p8);

    const p9 = try pigLatin(alloc, "are");
    defer alloc.free(p9);
    try testing.expectEqualStrings("areway", p9);
}

test "pig latin: null blank and no-vowel behavior" {
    const alloc = testing.allocator;

    const empty1 = try pigLatin(alloc, " ");
    defer alloc.free(empty1);
    try testing.expectEqualStrings("", empty1);

    const empty2 = try pigLatin(alloc, null);
    defer alloc.free(empty2);
    try testing.expectEqualStrings("", empty2);

    const no_vowel = try pigLatin(alloc, "rhythm");
    defer alloc.free(no_vowel);
    try testing.expectEqualStrings("mrhythay", no_vowel);
}

test "pig latin: extreme long input" {
    const alloc = testing.allocator;

    var long_word = [_]u8{'b'} ** 200_000;
    long_word[long_word.len - 1] = 'a';

    const out = try pigLatin(alloc, &long_word);
    defer alloc.free(out);

    try testing.expectEqual(@as(usize, 200_002), out.len);
    try testing.expect(out[0] == 'a');
    try testing.expect(out[out.len - 2] == 'a');
    try testing.expect(out[out.len - 1] == 'y');
}
