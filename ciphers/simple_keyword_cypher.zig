//! Simple Keyword Cipher - Zig implementation
//! Reference: https://github.com/TheAlgorithms/Python/blob/master/ciphers/simple_keyword_cypher.py

const std = @import("std");
const testing = std.testing;
const Allocator = std.mem.Allocator;

pub const SimpleKeywordError = error{ KeyTooLong, InvalidInternalIndex };

const LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ";

/// Removes duplicate alphabetic characters (case-sensitive to input) while preserving spaces.
/// Time complexity: O(n^2) worst-case due membership checks, Space complexity: O(n)
pub fn removeDuplicates(allocator: Allocator, key: []const u8) ![]u8 {
    var out = std.ArrayListUnmanaged(u8){};
    errdefer out.deinit(allocator);

    for (key) |ch| {
        if (ch == ' ') {
            try out.append(allocator, ch);
            continue;
        }

        if (!std.ascii.isAlphabetic(ch)) continue;
        if (std.mem.indexOfScalar(u8, out.items, ch) == null) {
            try out.append(allocator, ch);
        }
    }

    return try out.toOwnedSlice(allocator);
}

fn alphabetIndexLikePython(idx: i64) ?usize {
    if (idx >= 0 and idx < 26) return @intCast(idx);
    if (idx < 0 and idx >= -26) return @intCast(26 + idx);
    return null;
}

/// Creates keyword substitution map for A-Z.
/// Time complexity: O(26 * key_len), Space complexity: O(1)
pub fn createCipherMap(allocator: Allocator, key: []const u8) ![26]u8 {
    const upper = try allocator.alloc(u8, key.len);
    defer allocator.free(upper);
    for (key, 0..) |ch, i| upper[i] = std.ascii.toUpper(ch);

    const key_no_dups = try removeDuplicates(allocator, upper);
    defer allocator.free(key_no_dups);

    if (key_no_dups.len > 26) return SimpleKeywordError.KeyTooLong;

    var map = [_]u8{0} ** 26;

    var filled: usize = 0;
    while (filled < key_no_dups.len and filled < 26) : (filled += 1) {
        map[filled] = key_no_dups[filled];
    }

    var offset: i64 = @intCast(key_no_dups.len);

    var i = filled;
    while (i < 26) : (i += 1) {
        var idx = @as(i64, @intCast(i)) - offset;
        var alpha_idx = alphabetIndexLikePython(idx) orelse return SimpleKeywordError.InvalidInternalIndex;
        var ch = LETTERS[alpha_idx];

        while (std.mem.indexOfScalar(u8, key_no_dups, ch) != null) {
            offset -= 1;
            idx = @as(i64, @intCast(i)) - offset;
            alpha_idx = alphabetIndexLikePython(idx) orelse return SimpleKeywordError.InvalidInternalIndex;
            ch = LETTERS[alpha_idx];
        }

        map[i] = ch;
    }

    return map;
}

/// Enciphers message using keyword map.
/// Time complexity: O(n), Space complexity: O(n)
pub fn encipher(allocator: Allocator, message: []const u8, cipher_map: [26]u8) ![]u8 {
    const out = try allocator.alloc(u8, message.len);
    errdefer allocator.free(out);

    for (message, 0..) |raw, i| {
        const ch = std.ascii.toUpper(raw);
        if (ch >= 'A' and ch <= 'Z') {
            out[i] = cipher_map[ch - 'A'];
        } else {
            out[i] = ch;
        }
    }

    return out;
}

/// Deciphers message using reverse keyword map.
/// Time complexity: O(n), Space complexity: O(n)
pub fn decipher(allocator: Allocator, message: []const u8, cipher_map: [26]u8) ![]u8 {
    var rev = [_]u8{0} ** 26;
    for (LETTERS, 0..) |ch, i| {
        const mapped = cipher_map[i];
        if (mapped >= 'A' and mapped <= 'Z') rev[mapped - 'A'] = ch;
    }

    const out = try allocator.alloc(u8, message.len);
    errdefer allocator.free(out);

    for (message, 0..) |raw, i| {
        const ch = std.ascii.toUpper(raw);
        if (ch >= 'A' and ch <= 'Z') {
            out[i] = rev[ch - 'A'];
        } else {
            out[i] = ch;
        }
    }

    return out;
}

test "simple keyword: remove duplicates sample" {
    const alloc = testing.allocator;
    const out = try removeDuplicates(alloc, "Hello World!!");
    defer alloc.free(out);
    try testing.expectEqualStrings("Helo Wrd", out);
}

test "simple keyword: python encipher/decipher sample" {
    const alloc = testing.allocator;

    const map = try createCipherMap(alloc, "Goodbye!!");

    const enc = try encipher(alloc, "Hello World!!", map);
    defer alloc.free(enc);
    try testing.expectEqualStrings("CYJJM VMQJB!!", enc);

    const dec = try decipher(alloc, enc, map);
    defer alloc.free(dec);
    try testing.expectEqualStrings("HELLO WORLD!!", dec);
}

test "simple keyword: empty key identity and extreme" {
    const alloc = testing.allocator;

    const map = try createCipherMap(alloc, "");
    const plain = "ABCDEFGHIJKLMNOPQRSTUVWXYZ";
    const enc = try encipher(alloc, plain, map);
    defer alloc.free(enc);
    try testing.expectEqualStrings(plain, enc);

    const n: usize = 12000;
    const msg = try alloc.alloc(u8, n);
    defer alloc.free(msg);

    for (msg, 0..) |*ch, i| {
        ch.* = switch (i % 4) {
            0 => 'a',
            1 => 'Z',
            2 => '!',
            else => ' ',
        };
    }

    const map2 = try createCipherMap(alloc, "ALGORITHM");
    const out = try encipher(alloc, msg, map2);
    defer alloc.free(out);
    try testing.expectEqual(@as(usize, n), out.len);
}
