//! Bifid Cipher - Zig implementation
//! Reference: https://github.com/TheAlgorithms/Python/blob/master/ciphers/bifid.py

const std = @import("std");
const testing = std.testing;
const Allocator = std.mem.Allocator;

pub const BifidError = error{ InvalidCharacter, InvalidIndex };

const SQUARE = [_][5]u8{
    [_]u8{ 'a', 'b', 'c', 'd', 'e' },
    [_]u8{ 'f', 'g', 'h', 'i', 'k' },
    [_]u8{ 'l', 'm', 'n', 'o', 'p' },
    [_]u8{ 'q', 'r', 's', 't', 'u' },
    [_]u8{ 'v', 'w', 'x', 'y', 'z' },
};

/// Returns 1-based Polybius coordinates for a letter.
/// Time complexity: O(25), Space complexity: O(1)
pub fn letterToNumbers(letter_raw: u8) ?[2]u8 {
    var letter = std.ascii.toLower(letter_raw);
    if (letter == 'j') letter = 'i';

    for (SQUARE, 0..) |row, r| {
        for (row, 0..) |ch, c| {
            if (ch == letter) return .{ @intCast(r + 1), @intCast(c + 1) };
        }
    }

    return null;
}

/// Returns letter from 1-based Polybius coordinates.
/// Time complexity: O(1), Space complexity: O(1)
pub fn numbersToLetter(index1: u8, index2: u8) !u8 {
    if (index1 < 1 or index1 > 5 or index2 < 1 or index2 > 5) return BifidError.InvalidIndex;
    return SQUARE[index1 - 1][index2 - 1];
}

fn normalizeMessage(allocator: Allocator, message: []const u8) ![]u8 {
    var out = std.ArrayListUnmanaged(u8){};
    errdefer out.deinit(allocator);

    for (message) |raw| {
        var ch = std.ascii.toLower(raw);
        if (ch == ' ') continue;
        if (ch == 'j') ch = 'i';
        if (!std.ascii.isAlphabetic(ch)) return BifidError.InvalidCharacter;
        try out.append(allocator, ch);
    }

    return try out.toOwnedSlice(allocator);
}

/// Encodes message with Bifid cipher.
/// Time complexity: O(n), Space complexity: O(n)
pub fn encode(allocator: Allocator, message: []const u8) ![]u8 {
    const normalized = try normalizeMessage(allocator, message);
    defer allocator.free(normalized);

    const n = normalized.len;
    const rows = try allocator.alloc(u8, n);
    defer allocator.free(rows);
    const cols = try allocator.alloc(u8, n);
    defer allocator.free(cols);

    for (normalized, 0..) |ch, i| {
        const numbers = letterToNumbers(ch) orelse return BifidError.InvalidCharacter;
        rows[i] = numbers[0];
        cols[i] = numbers[1];
    }

    const out = try allocator.alloc(u8, n);
    errdefer allocator.free(out);

    for (0..n) |i| {
        const index1 = if (2 * i < n) rows[2 * i] else cols[2 * i - n];
        const index2 = if (2 * i + 1 < n) rows[2 * i + 1] else cols[2 * i + 1 - n];
        out[i] = try numbersToLetter(index1, index2);
    }

    return out;
}

/// Decodes Bifid message.
/// Time complexity: O(n), Space complexity: O(n)
pub fn decode(allocator: Allocator, message: []const u8) ![]u8 {
    const normalized = try normalizeMessage(allocator, message);
    defer allocator.free(normalized);

    const n = normalized.len;
    const first_step = try allocator.alloc(u8, 2 * n);
    defer allocator.free(first_step);

    for (normalized, 0..) |ch, i| {
        const numbers = letterToNumbers(ch) orelse return BifidError.InvalidCharacter;
        first_step[2 * i] = numbers[0];
        first_step[2 * i + 1] = numbers[1];
    }

    const out = try allocator.alloc(u8, n);
    errdefer allocator.free(out);

    for (0..n) |i| {
        out[i] = try numbersToLetter(first_step[i], first_step[n + i]);
    }

    return out;
}

test "bifid: polybius coordinate samples" {
    try testing.expectEqualSlices(u8, &[_]u8{ 1, 1 }, &(letterToNumbers('a').?));
    try testing.expectEqualSlices(u8, &[_]u8{ 4, 5 }, &(letterToNumbers('u').?));
    try testing.expectEqual(@as(u8, 'u'), try numbersToLetter(4, 5));
}

test "bifid: python encode/decode samples" {
    const alloc = testing.allocator;

    const enc = try encode(alloc, "testmessage");
    defer alloc.free(enc);
    try testing.expectEqualStrings("qtltbdxrxlk", enc);

    const enc2 = try encode(alloc, "Test Message");
    defer alloc.free(enc2);
    try testing.expectEqualStrings("qtltbdxrxlk", enc2);

    const dec = try decode(alloc, "qtltbdxrxlk");
    defer alloc.free(dec);
    try testing.expectEqualStrings("testmessage", dec);
}

test "bifid: i and j normalization + extreme" {
    const alloc = testing.allocator;

    const a = try encode(alloc, "test j");
    defer alloc.free(a);
    const b = try encode(alloc, "test i");
    defer alloc.free(b);
    try testing.expectEqualStrings(a, b);

    const n: usize = 8000;
    const msg = try alloc.alloc(u8, n);
    defer alloc.free(msg);
    for (msg, 0..) |*ch, i| ch.* = if (i % 3 == 0) 'a' else if (i % 3 == 1) 'j' else ' ';

    const enc = try encode(alloc, msg);
    defer alloc.free(enc);
    const dec = try decode(alloc, enc);
    defer alloc.free(dec);

    const norm = try normalizeMessage(alloc, msg);
    defer alloc.free(norm);
    try testing.expectEqualSlices(u8, norm, dec);
}
