//! Trifid Cipher - Zig implementation
//! Reference: https://github.com/TheAlgorithms/Python/blob/master/ciphers/trifid_cipher.py

const std = @import("std");
const testing = std.testing;
const Allocator = std.mem.Allocator;

pub const TrifidError = error{ InvalidAlphabetLength, InvalidMessageCharacter, EmptyPeriod, InvalidMapping };

const TEST_VALUES = [_][3]u8{
    .{ '1', '1', '1' }, .{ '1', '1', '2' }, .{ '1', '1', '3' }, .{ '1', '2', '1' }, .{ '1', '2', '2' }, .{ '1', '2', '3' },
    .{ '1', '3', '1' }, .{ '1', '3', '2' }, .{ '1', '3', '3' }, .{ '2', '1', '1' }, .{ '2', '1', '2' }, .{ '2', '1', '3' },
    .{ '2', '2', '1' }, .{ '2', '2', '2' }, .{ '2', '2', '3' }, .{ '2', '3', '1' }, .{ '2', '3', '2' }, .{ '2', '3', '3' },
    .{ '3', '1', '1' }, .{ '3', '1', '2' }, .{ '3', '1', '3' }, .{ '3', '2', '1' }, .{ '3', '2', '2' }, .{ '3', '2', '3' },
    .{ '3', '3', '1' }, .{ '3', '3', '2' }, .{ '3', '3', '3' },
};

const DEFAULT_ALPHABET = "ABCDEFGHIJKLMNOPQRSTUVWXYZ.";

const Mapping = struct {
    char_to_num: [256][3]u8,
    has_char: [256]bool,
    number_to_char: [3][3][3]u8,
    has_number: [3][3][3]bool,
};

fn prepareMapping(allocator: Allocator, message_input: []const u8, alphabet_input: []const u8) !struct { message: []u8, mapping: Mapping } {
    var alphabet_buf: [27]u8 = undefined;
    var alphabet_len: usize = 0;
    for (alphabet_input) |ch| {
        if (ch == ' ') continue;
        if (alphabet_len >= 27) return TrifidError.InvalidAlphabetLength;
        alphabet_buf[alphabet_len] = std.ascii.toUpper(ch);
        alphabet_len += 1;
    }

    if (alphabet_len != 27) return TrifidError.InvalidAlphabetLength;

    var message = std.ArrayListUnmanaged(u8){};
    errdefer message.deinit(allocator);
    for (message_input) |ch| {
        if (ch == ' ') continue;
        try message.append(allocator, std.ascii.toUpper(ch));
    }

    var mapping = Mapping{
        .char_to_num = undefined,
        .has_char = [_]bool{false} ** 256,
        .number_to_char = undefined,
        .has_number = [_][3][3]bool{[_][3]bool{[_]bool{false} ** 3} ** 3} ** 3,
    };

    for (0..27) |i| {
        const letter = alphabet_buf[i];
        const num = TEST_VALUES[i];
        mapping.char_to_num[letter] = num;
        mapping.has_char[letter] = true;

        const a = num[0] - '1';
        const b = num[1] - '1';
        const c = num[2] - '1';
        mapping.number_to_char[a][b][c] = letter;
        mapping.has_number[a][b][c] = true;
    }

    for (message.items) |ch| {
        if (!mapping.has_char[ch]) {
            return TrifidError.InvalidMessageCharacter;
        }
    }

    return .{ .message = try message.toOwnedSlice(allocator), .mapping = mapping };
}

fn encryptPart(allocator: Allocator, message_part: []const u8, mapping: *const Mapping) ![]u8 {
    var one = std.ArrayListUnmanaged(u8){};
    var two = std.ArrayListUnmanaged(u8){};
    var three = std.ArrayListUnmanaged(u8){};
    defer one.deinit(allocator);
    defer two.deinit(allocator);
    defer three.deinit(allocator);

    for (message_part) |ch| {
        const num = mapping.char_to_num[ch];
        try one.append(allocator, num[0]);
        try two.append(allocator, num[1]);
        try three.append(allocator, num[2]);
    }

    var out = std.ArrayListUnmanaged(u8){};
    errdefer out.deinit(allocator);
    try out.appendSlice(allocator, one.items);
    try out.appendSlice(allocator, two.items);
    try out.appendSlice(allocator, three.items);
    return try out.toOwnedSlice(allocator);
}

fn decryptPart(allocator: Allocator, message_part: []const u8, mapping: *const Mapping) !struct { []u8, []u8, []u8 } {
    var this_part = std.ArrayListUnmanaged(u8){};
    defer this_part.deinit(allocator);

    for (message_part) |ch| {
        const num = mapping.char_to_num[ch];
        try this_part.appendSlice(allocator, &num);
    }

    const n = message_part.len;
    const a = try allocator.alloc(u8, n);
    const b = try allocator.alloc(u8, n);
    const c = try allocator.alloc(u8, n);

    @memcpy(a, this_part.items[0..n]);
    @memcpy(b, this_part.items[n .. 2 * n]);
    @memcpy(c, this_part.items[2 * n .. 3 * n]);
    return .{ a, b, c };
}

/// Encrypts message with Trifid cipher.
/// Time complexity: O(n), Space complexity: O(n)
pub fn encryptMessage(allocator: Allocator, message_input: []const u8, alphabet_input: []const u8, period: usize) ![]u8 {
    if (period == 0) return TrifidError.EmptyPeriod;

    const prep = try prepareMapping(allocator, message_input, alphabet_input);
    defer allocator.free(prep.message);

    if (prep.message.len == 0) return try allocator.alloc(u8, 0);

    var encrypted_numeric = std.ArrayListUnmanaged(u8){};
    defer encrypted_numeric.deinit(allocator);

    var i: usize = 0;
    while (i < prep.message.len) : (i += period) {
        const part = prep.message[i..@min(i + period, prep.message.len)];
        const enc_part = try encryptPart(allocator, part, &prep.mapping);
        defer allocator.free(enc_part);
        try encrypted_numeric.appendSlice(allocator, enc_part);
    }

    var encrypted = std.ArrayListUnmanaged(u8){};
    errdefer encrypted.deinit(allocator);

    var j: usize = 0;
    while (j < encrypted_numeric.items.len) : (j += 3) {
        const a = encrypted_numeric.items[j] - '1';
        const b = encrypted_numeric.items[j + 1] - '1';
        const c = encrypted_numeric.items[j + 2] - '1';
        if (!prep.mapping.has_number[a][b][c]) return TrifidError.InvalidMapping;
        try encrypted.append(allocator, prep.mapping.number_to_char[a][b][c]);
    }

    return try encrypted.toOwnedSlice(allocator);
}

/// Decrypts Trifid ciphertext.
/// Time complexity: O(n), Space complexity: O(n)
pub fn decryptMessage(allocator: Allocator, message_input: []const u8, alphabet_input: []const u8, period: usize) ![]u8 {
    if (period == 0) return TrifidError.EmptyPeriod;

    const prep = try prepareMapping(allocator, message_input, alphabet_input);
    defer allocator.free(prep.message);

    if (prep.message.len == 0) return try allocator.alloc(u8, 0);

    var decoded_trigrams = std.ArrayListUnmanaged([3]u8){};
    defer decoded_trigrams.deinit(allocator);

    var i: usize = 0;
    while (i < prep.message.len) : (i += period) {
        const part = prep.message[i..@min(i + period, prep.message.len)];
        const groups = try decryptPart(allocator, part, &prep.mapping);
        defer allocator.free(groups[0]);
        defer allocator.free(groups[1]);
        defer allocator.free(groups[2]);

        for (0..groups[0].len) |k| {
            try decoded_trigrams.append(allocator, .{ groups[0][k], groups[1][k], groups[2][k] });
        }
    }

    const out = try allocator.alloc(u8, decoded_trigrams.items.len);
    errdefer allocator.free(out);

    for (decoded_trigrams.items, 0..) |tri, idx| {
        const a = tri[0] - '1';
        const b = tri[1] - '1';
        const c = tri[2] - '1';
        if (!prep.mapping.has_number[a][b][c]) return TrifidError.InvalidMapping;
        out[idx] = prep.mapping.number_to_char[a][b][c];
    }

    return out;
}

test "trifid: python samples" {
    const alloc = testing.allocator;

    const enc = try encryptMessage(alloc, "I am a boy", DEFAULT_ALPHABET, 5);
    defer alloc.free(enc);
    try testing.expectEqualStrings("BCDGBQY", enc);

    const dec = try decryptMessage(alloc, "BCDGBQY", DEFAULT_ALPHABET, 5);
    defer alloc.free(dec);
    try testing.expectEqualStrings("IAMABOY", dec);

    const enc2 = try encryptMessage(alloc, "   aide toi le c  iel      ta id  era    ", "FELIXMARDSTBCGHJKNOPQUVWYZ+", 5);
    defer alloc.free(enc2);
    try testing.expectEqualStrings("FMJFVOISSUFTFPUFEQQC", enc2);

    const dec2 = try decryptMessage(alloc, "FMJFVOISSUFTFPUFEQQC", "FELIXMARDSTBCGHJKNOPQUVWYZ+", 5);
    defer alloc.free(dec2);
    try testing.expectEqualStrings("AIDETOILECIELTAIDERA", dec2);
}

test "trifid: invalid inputs" {
    const alloc = testing.allocator;
    try testing.expectError(TrifidError.InvalidAlphabetLength, encryptMessage(alloc, "HELLO", "ABC", 5));
    try testing.expectError(TrifidError.InvalidMessageCharacter, encryptMessage(alloc, "HELLO?", DEFAULT_ALPHABET, 5));
    try testing.expectError(TrifidError.EmptyPeriod, encryptMessage(alloc, "HELLO", DEFAULT_ALPHABET, 0));
}

test "trifid: extreme long round trip" {
    const alloc = testing.allocator;

    const n: usize = 6000;
    const msg = try alloc.alloc(u8, n);
    defer alloc.free(msg);
    for (msg, 0..) |*ch, i| ch.* = if (i % 13 == 0) ' ' else @as(u8, @intCast('A' + (i % 26)));

    const enc = try encryptMessage(alloc, msg, DEFAULT_ALPHABET, 7);
    defer alloc.free(enc);
    const dec = try decryptMessage(alloc, enc, DEFAULT_ALPHABET, 7);
    defer alloc.free(dec);

    var normalized = std.ArrayListUnmanaged(u8){};
    defer normalized.deinit(alloc);
    for (msg) |ch| if (ch != ' ') try normalized.append(alloc, ch);

    try testing.expectEqualSlices(u8, normalized.items, dec);
}
