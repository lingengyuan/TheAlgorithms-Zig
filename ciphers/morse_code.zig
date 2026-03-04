//! Morse Code Cipher - Zig implementation
//! Reference: https://github.com/TheAlgorithms/Python/blob/master/ciphers/morse_code.py

const std = @import("std");
const testing = std.testing;
const Allocator = std.mem.Allocator;

pub const MorseError = error{ InvalidCharacter, InvalidToken };

const Entry = struct { ch: u8, code: []const u8 };

const TABLE = [_]Entry{
    .{ .ch = 'A', .code = ".-" },     .{ .ch = 'B', .code = "-..." },    .{ .ch = 'C', .code = "-.-." },   .{ .ch = 'D', .code = "-.." },    .{ .ch = 'E', .code = "." },
    .{ .ch = 'F', .code = "..-." },   .{ .ch = 'G', .code = "--." },     .{ .ch = 'H', .code = "...." },   .{ .ch = 'I', .code = ".." },     .{ .ch = 'J', .code = ".---" },
    .{ .ch = 'K', .code = "-.-" },    .{ .ch = 'L', .code = ".-.." },    .{ .ch = 'M', .code = "--" },     .{ .ch = 'N', .code = "-." },     .{ .ch = 'O', .code = "---" },
    .{ .ch = 'P', .code = ".--." },   .{ .ch = 'Q', .code = "--.-" },    .{ .ch = 'R', .code = ".-." },    .{ .ch = 'S', .code = "..." },    .{ .ch = 'T', .code = "-" },
    .{ .ch = 'U', .code = "..-" },    .{ .ch = 'V', .code = "...-" },    .{ .ch = 'W', .code = ".--" },    .{ .ch = 'X', .code = "-..-" },   .{ .ch = 'Y', .code = "-.--" },
    .{ .ch = 'Z', .code = "--.." },   .{ .ch = '1', .code = ".----" },   .{ .ch = '2', .code = "..---" },  .{ .ch = '3', .code = "...--" },  .{ .ch = '4', .code = "....-" },
    .{ .ch = '5', .code = "....." },  .{ .ch = '6', .code = "-...." },   .{ .ch = '7', .code = "--..." },  .{ .ch = '8', .code = "---.." },  .{ .ch = '9', .code = "----." },
    .{ .ch = '0', .code = "-----" },  .{ .ch = '&', .code = ".-..." },   .{ .ch = '@', .code = ".--.-." }, .{ .ch = ':', .code = "---..." }, .{ .ch = ',', .code = "--..--" },
    .{ .ch = '.', .code = ".-.-.-" }, .{ .ch = '\'', .code = ".----." }, .{ .ch = '"', .code = ".-..-." }, .{ .ch = '?', .code = "..--.." }, .{ .ch = '/', .code = "-..-." },
    .{ .ch = '=', .code = "-...-" },  .{ .ch = '+', .code = ".-.-." },   .{ .ch = '-', .code = "-....-" }, .{ .ch = '(', .code = "-.--." },  .{ .ch = ')', .code = "-.--.-" },
    .{ .ch = '!', .code = "-.-.--" }, .{ .ch = ' ', .code = "/" },
};

fn findCode(ch: u8) ?[]const u8 {
    for (TABLE) |e| {
        if (e.ch == ch) return e.code;
    }
    return null;
}

fn findChar(code: []const u8) ?u8 {
    for (TABLE) |e| {
        if (std.mem.eql(u8, e.code, code)) return e.ch;
    }
    return null;
}

/// Encrypts text to Morse code separated by single spaces.
/// Time complexity: O(n * table), Space complexity: O(n)
pub fn encrypt(allocator: Allocator, message: []const u8) ![]u8 {
    var out = std.ArrayListUnmanaged(u8){};
    errdefer out.deinit(allocator);

    for (message, 0..) |ch, i| {
        const upper = std.ascii.toUpper(ch);
        const code = findCode(upper) orelse return MorseError.InvalidCharacter;
        if (i != 0) try out.append(allocator, ' ');
        try out.appendSlice(allocator, code);
    }

    return try out.toOwnedSlice(allocator);
}

/// Decrypts Morse code tokens separated by spaces.
/// Time complexity: O(n * table), Space complexity: O(n)
pub fn decrypt(allocator: Allocator, message: []const u8) ![]u8 {
    var out = std.ArrayListUnmanaged(u8){};
    errdefer out.deinit(allocator);

    var it = std.mem.tokenizeScalar(u8, message, ' ');
    while (it.next()) |token| {
        const ch = findChar(token) orelse return MorseError.InvalidToken;
        try out.append(allocator, ch);
    }

    return try out.toOwnedSlice(allocator);
}

test "morse code: python samples" {
    const alloc = testing.allocator;

    const enc = try encrypt(alloc, "Sos!");
    defer alloc.free(enc);
    try testing.expectEqualStrings("... --- ... -.-.--", enc);

    const dec = try decrypt(alloc, "... --- ... -.-.--");
    defer alloc.free(dec);
    try testing.expectEqualStrings("SOS!", dec);
}

test "morse code: case-insensitive encrypt" {
    const alloc = testing.allocator;
    const a = try encrypt(alloc, "SOS!");
    defer alloc.free(a);
    const b = try encrypt(alloc, "sos!");
    defer alloc.free(b);
    try testing.expectEqualStrings(a, b);
}

test "morse code: invalid inputs" {
    const alloc = testing.allocator;
    try testing.expectError(MorseError.InvalidCharacter, encrypt(alloc, "你好"));
    try testing.expectError(MorseError.InvalidToken, decrypt(alloc, "... --- ... ....--.-"));
}

test "morse code: extreme long round-trip" {
    const alloc = testing.allocator;
    const text = "THE ALGORITHMS 2026! THE ALGORITHMS 2026!";

    const enc = try encrypt(alloc, text);
    defer alloc.free(enc);
    const dec = try decrypt(alloc, enc);
    defer alloc.free(dec);

    try testing.expectEqualStrings(text, dec);
}
