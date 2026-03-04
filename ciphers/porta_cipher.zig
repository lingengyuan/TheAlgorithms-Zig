//! Porta Cipher - Zig implementation
//! Reference: https://github.com/TheAlgorithms/Python/blob/master/ciphers/porta_cipher.py

const std = @import("std");
const testing = std.testing;
const Allocator = std.mem.Allocator;

pub const PortaError = error{
    EmptyKey,
    InvalidKeyCharacter,
    InvalidTextCharacter,
};

const ROW_A = "ABCDEFGHIJKLM";
const ROW_B_BASE = "NOPQRSTUVWXYZ";

pub const PortaTableRow = struct {
    row0: [13]u8,
    row1: [13]u8,
};

fn shiftGroupForKeyChar(ch: u8) ?usize {
    const upper = std.ascii.toUpper(ch);
    if (upper < 'A' or upper > 'Z') return null;
    return (upper - 'A') / 2;
}

fn buildRow(group: usize) [13]u8 {
    var out: [13]u8 = undefined;
    const g = group % 13;

    for (0..13) |i| {
        const src = (i + 13 - g) % 13;
        out[i] = ROW_B_BASE[src];
    }

    return out;
}

/// Generates Porta substitution table rows for each key character.
/// Time complexity: O(k), Space complexity: O(k)
pub fn generateTable(allocator: Allocator, key: []const u8) ![]PortaTableRow {
    if (key.len == 0) return PortaError.EmptyKey;

    const table = try allocator.alloc(PortaTableRow, key.len);
    errdefer allocator.free(table);

    for (key, 0..) |ch, i| {
        const group = shiftGroupForKeyChar(ch) orelse return PortaError.InvalidKeyCharacter;
        table[i] = PortaTableRow{ .row0 = ROW_A.*, .row1 = buildRow(group) };
    }

    return table;
}

fn getOpponent(row: PortaTableRow, ch: u8) PortaError!u8 {
    const upper = std.ascii.toUpper(ch);

    if (std.mem.indexOfScalar(u8, &row.row0, upper)) |col| {
        return row.row1[col];
    }
    if (std.mem.indexOfScalar(u8, &row.row1, upper)) |col| {
        return row.row0[col];
    }

    return PortaError.InvalidTextCharacter;
}

/// Encrypts text with Porta cipher.
/// Time complexity: O(n), Space complexity: O(n)
pub fn encrypt(allocator: Allocator, key: []const u8, words: []const u8) ![]u8 {
    const table = try generateTable(allocator, key);
    defer allocator.free(table);

    const out = try allocator.alloc(u8, words.len);
    errdefer allocator.free(out);

    var count: usize = 0;
    for (words, 0..) |ch, i| {
        out[i] = try getOpponent(table[count], ch);
        count = (count + 1) % table.len;
    }

    return out;
}

/// Decrypts text with Porta cipher.
/// Porta cipher is reciprocal, so decrypt == encrypt.
/// Time complexity: O(n), Space complexity: O(n)
pub fn decrypt(allocator: Allocator, key: []const u8, words: []const u8) ![]u8 {
    return encrypt(allocator, key, words);
}

test "porta: python samples" {
    const alloc = testing.allocator;

    const table = try generateTable(alloc, "marvin");
    defer alloc.free(table);
    try testing.expectEqualStrings("UVWXYZNOPQRST", &table[0].row1);

    const enc = try encrypt(alloc, "marvin", "jessica");
    defer alloc.free(enc);
    try testing.expectEqualStrings("QRACRWU", enc);

    const dec = try decrypt(alloc, "marvin", "QRACRWU");
    defer alloc.free(dec);
    try testing.expectEqualStrings("JESSICA", dec);
}

test "porta: invalid cases" {
    const alloc = testing.allocator;

    try testing.expectError(PortaError.EmptyKey, encrypt(alloc, "", "HELLO"));
    try testing.expectError(PortaError.InvalidKeyCharacter, encrypt(alloc, "K3Y", "HELLO"));
    try testing.expectError(PortaError.InvalidTextCharacter, encrypt(alloc, "KEY", "HELLO!"));
}

test "porta: extreme long round trip" {
    const alloc = testing.allocator;
    const n: usize = 10000;

    const text = try alloc.alloc(u8, n);
    defer alloc.free(text);

    for (text, 0..) |*ch, i| {
        ch.* = @as(u8, @intCast('A' + (i % 26)));
    }

    const enc = try encrypt(alloc, "ALGORITHMS", text);
    defer alloc.free(enc);

    const dec = try decrypt(alloc, "ALGORITHMS", enc);
    defer alloc.free(dec);

    try testing.expectEqualSlices(u8, text, dec);
}
