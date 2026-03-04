//! Polybius Square Cipher - Zig implementation
//! Reference: https://github.com/TheAlgorithms/Python/blob/master/ciphers/polybius.py

const std = @import("std");
const testing = std.testing;
const Allocator = std.mem.Allocator;

pub const PolybiusError = error{
    InvalidCharacter,
    InvalidCode,
};

const SQUARE = [_][5]u8{
    [_]u8{ 'a', 'b', 'c', 'd', 'e' },
    [_]u8{ 'f', 'g', 'h', 'i', 'k' },
    [_]u8{ 'l', 'm', 'n', 'o', 'p' },
    [_]u8{ 'q', 'r', 's', 't', 'u' },
    [_]u8{ 'v', 'w', 'x', 'y', 'z' },
};

fn letterToNumbers(letter: u8) ?[2]u8 {
    for (SQUARE, 0..) |row, r| {
        for (row, 0..) |ch, c| {
            if (ch == letter) return .{ @intCast(r + 1), @intCast(c + 1) };
        }
    }
    return null;
}

fn numbersToLetter(r: u8, c: u8) ?u8 {
    if (r < 1 or r > 5 or c < 1 or c > 5) return null;
    return SQUARE[r - 1][c - 1];
}

/// Encodes message with Polybius square.
/// Converts to lowercase and maps 'j' -> 'i', preserving spaces.
/// Time complexity: O(n), Space complexity: O(n)
pub fn encode(allocator: Allocator, message: []const u8) ![]u8 {
    var out = std.ArrayListUnmanaged(u8){};
    errdefer out.deinit(allocator);

    for (message) |ch| {
        if (ch == ' ') {
            try out.append(allocator, ' ');
            continue;
        }

        var lower = std.ascii.toLower(ch);
        if (lower == 'j') lower = 'i';

        const nums = letterToNumbers(lower) orelse return PolybiusError.InvalidCharacter;
        try out.append(allocator, @intCast('0' + nums[0]));
        try out.append(allocator, @intCast('0' + nums[1]));
    }

    return try out.toOwnedSlice(allocator);
}

/// Decodes Polybius digits back to letters.
/// Mirrors Python behavior of doubling spaces before pair parsing.
/// Time complexity: O(n), Space complexity: O(n)
pub fn decode(allocator: Allocator, message: []const u8) ![]u8 {
    var doubled = std.ArrayListUnmanaged(u8){};
    defer doubled.deinit(allocator);

    for (message) |ch| {
        if (ch == ' ') {
            try doubled.appendSlice(allocator, "  ");
        } else {
            try doubled.append(allocator, ch);
        }
    }

    var out = std.ArrayListUnmanaged(u8){};
    errdefer out.deinit(allocator);

    const pair_count = doubled.items.len / 2;
    for (0..pair_count) |i| {
        const a = doubled.items[i * 2];
        const b = doubled.items[i * 2 + 1];

        if (a == ' ') {
            try out.append(allocator, ' ');
            continue;
        }

        if (!(a >= '0' and a <= '9' and b >= '0' and b <= '9')) return PolybiusError.InvalidCode;
        const r: u8 = @intCast(a - '0');
        const c: u8 = @intCast(b - '0');
        const letter = numbersToLetter(r, c) orelse return PolybiusError.InvalidCode;
        try out.append(allocator, letter);
    }

    return try out.toOwnedSlice(allocator);
}

test "polybius: python samples" {
    const alloc = testing.allocator;

    const enc = try encode(alloc, "test message");
    defer alloc.free(enc);
    try testing.expectEqualStrings("44154344 32154343112215", enc);

    const enc2 = try encode(alloc, "Test Message");
    defer alloc.free(enc2);
    try testing.expectEqualStrings("44154344 32154343112215", enc2);

    const dec = try decode(alloc, "44154344 32154343112215");
    defer alloc.free(dec);
    try testing.expectEqualStrings("test message", dec);
}

test "polybius: j maps to i" {
    const alloc = testing.allocator;
    const enc = try encode(alloc, "j");
    defer alloc.free(enc);
    try testing.expectEqualStrings("24", enc);
}

test "polybius: invalid inputs" {
    const alloc = testing.allocator;
    try testing.expectError(PolybiusError.InvalidCharacter, encode(alloc, "test!"));
    try testing.expectError(PolybiusError.InvalidCode, decode(alloc, "9911"));
}

test "polybius: extreme round-trip" {
    const alloc = testing.allocator;
    const n: usize = 6000;
    const text = try alloc.alloc(u8, n);
    defer alloc.free(text);
    for (0..n) |i| {
        text[i] = if (i % 9 == 0) ' ' else @intCast('a' + (i % 26));
    }

    const enc = try encode(alloc, text);
    defer alloc.free(enc);
    const dec = try decode(alloc, enc);
    defer alloc.free(dec);

    // Python maps j -> i, so normalize for equality.
    const expected = try alloc.alloc(u8, n);
    defer alloc.free(expected);
    for (text, 0..) |ch, i| {
        var c = std.ascii.toLower(ch);
        if (c == 'j') c = 'i';
        expected[i] = c;
    }

    try testing.expectEqualSlices(u8, expected, dec);
}
