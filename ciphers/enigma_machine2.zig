//! Enigma Machine 2 - Zig implementation
//! Reference: https://github.com/TheAlgorithms/Python/blob/master/ciphers/enigma_machine2.py

const std = @import("std");
const testing = std.testing;
const Allocator = std.mem.Allocator;

pub const EnigmaError = error{
    NonUniqueRotors,
    InvalidRotorPosition,
    OddPlugboardSymbols,
    InvalidPlugboardCharacter,
    DuplicatePlugboardCharacter,
};

const ABC = "ABCDEFGHIJKLMNOPQRSTUVWXYZ";

pub const rotor1 = "EGZWVONAHDCLFQMSIPJBYUKXTR";
pub const rotor2 = "FOBHMDKEXQNRAULPGSJVTYICZW";
pub const rotor3 = "ZJXESIUQLHAVRMDOYGTNFWPBKC";
pub const rotor4 = "RMDJXFUWGISLHVTCQNKYPBEZOA";
pub const rotor5 = "SGLCPQWZHKXAREONTFBVIYJUDM";
pub const rotor6 = "HVSICLTYKQUBXDWAJZOMFGPREN";
pub const rotor7 = "RZWQHFMVDBKICJLNTUXAGYPSOE";
pub const rotor8 = "LFKIJODBEGAMQPXVUHYSTCZRWN";
pub const rotor9 = "KOAEGVDHXPQZMLFTYWJNBRCIUS";

const Plugboard = struct {
    has: [26]bool,
    map: [26]u8,

    fn empty() Plugboard {
        return Plugboard{
            .has = [_]bool{false} ** 26,
            .map = [_]u8{0} ** 26,
        };
    }

    fn apply(self: Plugboard, ch: u8) u8 {
        const idx = ch - 'A';
        return if (self.has[idx]) self.map[idx] else ch;
    }
};

fn parsePlugboard(pbstring: []const u8) !Plugboard {
    if ((pbstring.len & 1) != 0) return EnigmaError.OddPlugboardSymbols;
    if (pbstring.len == 0) return Plugboard.empty();

    var pb = Plugboard.empty();
    var used = [_]bool{false} ** 26;

    for (pbstring) |ch| {
        if (ch < 'A' or ch > 'Z') return EnigmaError.InvalidPlugboardCharacter;
        const idx = ch - 'A';
        if (used[idx]) return EnigmaError.DuplicatePlugboardCharacter;
        used[idx] = true;
    }

    var i: usize = 0;
    while (i + 1 < pbstring.len) : (i += 2) {
        const a = pbstring[i];
        const b = pbstring[i + 1];
        pb.has[a - 'A'] = true;
        pb.has[b - 'A'] = true;
        pb.map[a - 'A'] = b;
        pb.map[b - 'A'] = a;
    }

    return pb;
}

fn validate(rotor_position: [3]u8, rotor_selection: [3][]const u8, plugb: []const u8) !Plugboard {
    // 3 unique rotors by content.
    if (std.mem.eql(u8, rotor_selection[0], rotor_selection[1]) or
        std.mem.eql(u8, rotor_selection[0], rotor_selection[2]) or
        std.mem.eql(u8, rotor_selection[1], rotor_selection[2]))
    {
        return EnigmaError.NonUniqueRotors;
    }

    for (rotor_position) |pos| {
        if (pos == 0 or pos > 26) return EnigmaError.InvalidRotorPosition;
    }

    return parsePlugboard(plugb);
}

fn mod26(value: i32) usize {
    return @intCast(@mod(value, 26));
}

/// Enigma encryption/decryption (same operation).
/// Time complexity: O(n * 26), Space complexity: O(n)
pub fn enigma(
    allocator: Allocator,
    text_input: []const u8,
    rotor_position: [3]u8,
    rotor_selection: [3][]const u8,
    plugb_input: []const u8,
) ![]u8 {
    const text = try allocator.alloc(u8, text_input.len);
    defer allocator.free(text);
    for (text_input, 0..) |ch, i| text[i] = std.ascii.toUpper(ch);

    const pb_upper = try allocator.alloc(u8, plugb_input.len);
    defer allocator.free(pb_upper);
    for (plugb_input, 0..) |ch, i| pb_upper[i] = std.ascii.toUpper(ch);

    const plugboard = try validate(rotor_position, rotor_selection, pb_upper);

    var rotorpos1: i32 = @as(i32, rotor_position[0]) - 1;
    var rotorpos2: i32 = @as(i32, rotor_position[1]) - 1;
    var rotorpos3: i32 = @as(i32, rotor_position[2]) - 1;

    const r1 = rotor_selection[0];
    const r2 = rotor_selection[1];
    const r3 = rotor_selection[2];

    const out = try allocator.alloc(u8, text.len);
    errdefer allocator.free(out);

    for (text, 0..) |symbol_raw, i| {
        var symbol = symbol_raw;

        if (std.mem.indexOfScalar(u8, ABC, symbol) != null) {
            symbol = plugboard.apply(symbol);

            var index = @as(i32, @intCast(std.mem.indexOfScalar(u8, ABC, symbol).?)) + rotorpos1;
            symbol = r1[mod26(index)];

            index = @as(i32, @intCast(std.mem.indexOfScalar(u8, ABC, symbol).?)) + rotorpos2;
            symbol = r2[mod26(index)];

            index = @as(i32, @intCast(std.mem.indexOfScalar(u8, ABC, symbol).?)) + rotorpos3;
            symbol = r3[mod26(index)];

            // Reflector (rot13 mapping)
            const ridx = std.mem.indexOfScalar(u8, ABC, symbol).?;
            symbol = ABC[(ridx + 13) % 26];

            symbol = ABC[mod26(@as(i32, @intCast(std.mem.indexOfScalar(u8, r3, symbol).?)) - rotorpos3)];
            symbol = ABC[mod26(@as(i32, @intCast(std.mem.indexOfScalar(u8, r2, symbol).?)) - rotorpos2)];
            symbol = ABC[mod26(@as(i32, @intCast(std.mem.indexOfScalar(u8, r1, symbol).?)) - rotorpos1)];

            symbol = plugboard.apply(symbol);

            // Rotor stepping.
            rotorpos1 += 1;
            if (rotorpos1 >= 26) {
                rotorpos1 = 0;
                rotorpos2 += 1;
            }
            if (rotorpos2 >= 26) {
                rotorpos2 = 0;
                rotorpos3 += 1;
            }
            if (rotorpos3 >= 26) rotorpos3 = 0;
        }

        out[i] = symbol;
    }

    return out;
}

test "enigma: python doctest samples" {
    const alloc = testing.allocator;

    const a = try enigma(alloc, "Hello World!", .{ 1, 2, 1 }, .{ rotor1, rotor2, rotor3 }, "pictures");
    defer alloc.free(a);
    try testing.expectEqualStrings("KORYH JUHHI!", a);

    const b = try enigma(alloc, "KORYH, juhhi!", .{ 1, 2, 1 }, .{ rotor1, rotor2, rotor3 }, "pictures");
    defer alloc.free(b);
    try testing.expectEqualStrings("HELLO, WORLD!", b);

    const c = try enigma(alloc, "hello world!", .{ 1, 1, 1 }, .{ rotor1, rotor2, rotor3 }, "pictures");
    defer alloc.free(c);
    try testing.expectEqualStrings("FPNCZ QWOBU!", c);

    const d = try enigma(alloc, "FPNCZ QWOBU", .{ 1, 1, 1 }, .{ rotor1, rotor2, rotor3 }, "pictures");
    defer alloc.free(d);
    try testing.expectEqualStrings("HELLO WORLD", d);
}

test "enigma: invalid config" {
    const alloc = testing.allocator;

    try testing.expectError(EnigmaError.NonUniqueRotors, enigma(alloc, "ABC", .{ 1, 1, 1 }, .{ rotor1, rotor1, rotor3 }, ""));
    try testing.expectError(EnigmaError.InvalidRotorPosition, enigma(alloc, "ABC", .{ 0, 1, 1 }, .{ rotor1, rotor2, rotor3 }, ""));
    try testing.expectError(EnigmaError.OddPlugboardSymbols, enigma(alloc, "ABC", .{ 1, 1, 1 }, .{ rotor1, rotor2, rotor3 }, "ABC"));
}

test "enigma: extreme long round trip" {
    const alloc = testing.allocator;

    const n: usize = 8000;
    const text = try alloc.alloc(u8, n);
    defer alloc.free(text);

    for (text, 0..) |*ch, i| {
        ch.* = switch (i % 6) {
            0 => 'A',
            1 => 'z',
            2 => ' ',
            3 => ',',
            4 => '!',
            else => 'M',
        };
    }

    const enc = try enigma(alloc, text, .{ 1, 1, 1 }, .{ rotor2, rotor4, rotor8 }, "PICTURES");
    defer alloc.free(enc);
    const dec = try enigma(alloc, enc, .{ 1, 1, 1 }, .{ rotor2, rotor4, rotor8 }, "PICTURES");
    defer alloc.free(dec);

    var upper = try alloc.alloc(u8, text.len);
    defer alloc.free(upper);
    for (text, 0..) |ch, i| upper[i] = std.ascii.toUpper(ch);

    try testing.expectEqualSlices(u8, upper, dec);
}
