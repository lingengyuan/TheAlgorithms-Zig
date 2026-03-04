//! Playfair Cipher - Zig implementation
//! Reference: https://github.com/TheAlgorithms/Python/blob/master/ciphers/playfair_cipher.py

const std = @import("std");
const testing = std.testing;
const Allocator = std.mem.Allocator;

pub const PlayfairError = error{InvalidCharacter};

const ALPHABET = "ABCDEFGHIKLMNOPQRSTUVWXYZ";

/// Prepares plaintext: keep letters, uppercase, insert X between repeats, ensure even length.
/// Time complexity: O(n), Space complexity: O(n)
pub fn prepareInput(allocator: Allocator, dirty: []const u8) ![]u8 {
    var cleaned = std.ArrayListUnmanaged(u8){};
    defer cleaned.deinit(allocator);

    for (dirty) |ch| {
        if (!std.ascii.isAlphabetic(ch)) continue;
        var up = std.ascii.toUpper(ch);
        if (up == 'J') up = 'I';
        try cleaned.append(allocator, up);
    }

    if (cleaned.items.len < 2) {
        const out = try allocator.alloc(u8, cleaned.items.len + (if (cleaned.items.len == 1) @as(usize, 1) else @as(usize, 0)));
        if (cleaned.items.len == 1) {
            out[0] = cleaned.items[0];
            out[1] = 'X';
        }
        return out;
    }

    var out = std.ArrayListUnmanaged(u8){};
    errdefer out.deinit(allocator);

    var i: usize = 0;
    while (i + 1 < cleaned.items.len) : (i += 1) {
        try out.append(allocator, cleaned.items[i]);
        if (cleaned.items[i] == cleaned.items[i + 1]) {
            try out.append(allocator, 'X');
        }
    }
    try out.append(allocator, cleaned.items[cleaned.items.len - 1]);

    if ((out.items.len & 1) == 1) try out.append(allocator, 'X');

    return try out.toOwnedSlice(allocator);
}

/// Generates 5x5 Playfair table as 25-byte flat array.
/// Time complexity: O(len(key) + 25), Space complexity: O(1)
pub fn generateTable(key: []const u8) [25]u8 {
    var table = [_]u8{0} ** 25;
    var used = [_]bool{false} ** 26;
    used['J' - 'A'] = true; // J omitted in Playfair table.

    var idx: usize = 0;

    for (key) |raw| {
        var ch = std.ascii.toUpper(raw);
        if (ch == 'J') ch = 'I';
        if (ch < 'A' or ch > 'Z') continue;
        const id = ch - 'A';
        if (!used[id]) {
            used[id] = true;
            table[idx] = ch;
            idx += 1;
        }
    }

    for (ALPHABET) |ch| {
        const id = ch - 'A';
        if (!used[id]) {
            used[id] = true;
            table[idx] = ch;
            idx += 1;
        }
    }

    return table;
}

fn indexInTable(table: *const [25]u8, ch_raw: u8) ?usize {
    var ch = std.ascii.toUpper(ch_raw);
    if (ch == 'J') ch = 'I';
    return std.mem.indexOfScalar(u8, table, ch);
}

/// Encodes plaintext with Playfair cipher.
/// Time complexity: O(n), Space complexity: O(n)
pub fn encode(allocator: Allocator, plaintext: []const u8, key: []const u8) ![]u8 {
    const table = generateTable(key);
    const clean = try prepareInput(allocator, plaintext);
    defer allocator.free(clean);

    const out = try allocator.alloc(u8, clean.len);
    errdefer allocator.free(out);

    var i: usize = 0;
    while (i < clean.len) : (i += 2) {
        const char1 = clean[i];
        const char2 = clean[i + 1];

        const idx1 = indexInTable(&table, char1) orelse return PlayfairError.InvalidCharacter;
        const idx2 = indexInTable(&table, char2) orelse return PlayfairError.InvalidCharacter;

        const row1 = idx1 / 5;
        const col1 = idx1 % 5;
        const row2 = idx2 / 5;
        const col2 = idx2 % 5;

        if (row1 == row2) {
            out[i] = table[row1 * 5 + ((col1 + 1) % 5)];
            out[i + 1] = table[row2 * 5 + ((col2 + 1) % 5)];
        } else if (col1 == col2) {
            out[i] = table[((row1 + 1) % 5) * 5 + col1];
            out[i + 1] = table[((row2 + 1) % 5) * 5 + col2];
        } else {
            out[i] = table[row1 * 5 + col2];
            out[i + 1] = table[row2 * 5 + col1];
        }
    }

    return out;
}

/// Decodes ciphertext with Playfair cipher.
/// Time complexity: O(n), Space complexity: O(n)
pub fn decode(allocator: Allocator, ciphertext: []const u8, key: []const u8) ![]u8 {
    const table = generateTable(key);

    const out = try allocator.alloc(u8, ciphertext.len);
    errdefer allocator.free(out);

    var i: usize = 0;
    while (i + 1 < ciphertext.len) : (i += 2) {
        const idx1 = indexInTable(&table, ciphertext[i]) orelse return PlayfairError.InvalidCharacter;
        const idx2 = indexInTable(&table, ciphertext[i + 1]) orelse return PlayfairError.InvalidCharacter;

        const row1 = idx1 / 5;
        const col1 = idx1 % 5;
        const row2 = idx2 / 5;
        const col2 = idx2 % 5;

        if (row1 == row2) {
            out[i] = table[row1 * 5 + ((col1 + 4) % 5)];
            out[i + 1] = table[row2 * 5 + ((col2 + 4) % 5)];
        } else if (col1 == col2) {
            out[i] = table[((row1 + 4) % 5) * 5 + col1];
            out[i + 1] = table[((row2 + 4) % 5) * 5 + col2];
        } else {
            out[i] = table[row1 * 5 + col2];
            out[i + 1] = table[row2 * 5 + col1];
        }
    }

    return out;
}

test "playfair: python encode samples" {
    const alloc = testing.allocator;

    const a = try encode(alloc, "Hello", "MONARCHY");
    defer alloc.free(a);
    try testing.expectEqualStrings("CFSUPM", a);

    const b = try encode(alloc, "attack on the left flank", "EMERGENCY");
    defer alloc.free(b);
    try testing.expectEqualStrings("DQZSBYFSDZFMFNLOHFDRSG", b);
}

test "playfair: python decode samples" {
    const alloc = testing.allocator;

    const a = try decode(alloc, "BMZFAZRZDH", "HAZARD");
    defer alloc.free(a);
    try testing.expectEqualStrings("FIREHAZARD", a);

    const b = try decode(alloc, "HNBWBPQT", "AUTOMOBILE");
    defer alloc.free(b);
    try testing.expectEqualStrings("DRIVINGX", b);
}

test "playfair: prepare input and extreme round trip" {
    const alloc = testing.allocator;

    const prepared = try prepareInput(alloc, "balloon");
    defer alloc.free(prepared);
    try testing.expectEqualStrings("BALXLOXONX", prepared);

    const n: usize = 9000;
    const plain = try alloc.alloc(u8, n);
    defer alloc.free(plain);
    for (plain, 0..) |*ch, i| ch.* = if (i % 11 == 0) ' ' else @as(u8, @intCast('A' + (i % 26)));

    const enc = try encode(alloc, plain, "GREETING");
    defer alloc.free(enc);
    const dec = try decode(alloc, enc, "GREETING");
    defer alloc.free(dec);

    const norm = try prepareInput(alloc, plain);
    defer alloc.free(norm);
    try testing.expectEqualSlices(u8, norm, dec);
}
