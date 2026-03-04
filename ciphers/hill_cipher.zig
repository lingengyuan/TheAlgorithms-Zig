//! Hill Cipher - Zig implementation
//! Reference: https://github.com/TheAlgorithms/Python/blob/master/ciphers/hill_cipher.py

const std = @import("std");
const testing = std.testing;
const Allocator = std.mem.Allocator;

pub const HillCipherError = error{ InvalidDeterminant, UnsupportedMatrixOrder, InvalidCharacter };

const KEY_STRING = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789";
const MODULUS: i64 = 36;

pub const HillCipher = struct {
    encrypt_key: [2][2]i64,
    break_key: usize,

    pub fn init(encrypt_key_input: [2][2]i64) !HillCipher {
        var key = encrypt_key_input;
        for (0..2) |r| {
            for (0..2) |c| {
                key[r][c] = @mod(key[r][c], MODULUS);
            }
        }

        const det = @mod(key[0][0] * key[1][1] - key[0][1] * key[1][0], MODULUS);
        if (gcd(det, MODULUS) != 1) return HillCipherError.InvalidDeterminant;

        return HillCipher{ .encrypt_key = key, .break_key = 2 };
    }

    pub fn replaceLetters(self: HillCipher, letter: u8) !usize {
        _ = self;
        return std.mem.indexOfScalar(u8, KEY_STRING, std.ascii.toUpper(letter)) orelse HillCipherError.InvalidCharacter;
    }

    pub fn replaceDigits(self: HillCipher, num_raw: i64) u8 {
        _ = self;
        const num: usize = @intCast(@mod(num_raw, MODULUS));
        return KEY_STRING[num];
    }

    pub fn processText(self: HillCipher, allocator: Allocator, text: []const u8) ![]u8 {
        var chars = std.ArrayListUnmanaged(u8){};
        errdefer chars.deinit(allocator);

        for (text) |raw| {
            const ch = std.ascii.toUpper(raw);
            if (std.mem.indexOfScalar(u8, KEY_STRING, ch) != null) {
                try chars.append(allocator, ch);
            }
        }

        if (chars.items.len == 0) return try chars.toOwnedSlice(allocator);

        const last = chars.items[chars.items.len - 1];
        while ((chars.items.len % self.break_key) != 0) {
            try chars.append(allocator, last);
        }

        return try chars.toOwnedSlice(allocator);
    }

    pub fn encrypt(self: HillCipher, allocator: Allocator, text: []const u8) ![]u8 {
        const clean = try self.processText(allocator, text);
        defer allocator.free(clean);

        const out = try allocator.alloc(u8, clean.len);
        errdefer allocator.free(out);

        var i: usize = 0;
        while (i < clean.len) : (i += 2) {
            const v0: i64 = @intCast(try self.replaceLetters(clean[i]));
            const v1: i64 = @intCast(try self.replaceLetters(clean[i + 1]));

            const e0 = @mod(self.encrypt_key[0][0] * v0 + self.encrypt_key[0][1] * v1, MODULUS);
            const e1 = @mod(self.encrypt_key[1][0] * v0 + self.encrypt_key[1][1] * v1, MODULUS);

            out[i] = self.replaceDigits(e0);
            out[i + 1] = self.replaceDigits(e1);
        }

        return out;
    }

    pub fn makeDecryptKey(self: HillCipher) ![2][2]i64 {
        const a = self.encrypt_key[0][0];
        const b = self.encrypt_key[0][1];
        const c = self.encrypt_key[1][0];
        const d = self.encrypt_key[1][1];

        var det = @mod(a * d - b * c, MODULUS);
        if (det < 0) det += MODULUS;

        var det_inv: ?i64 = null;
        for (0..36) |i| {
            if (@mod(det * @as(i64, @intCast(i)), MODULUS) == 1) {
                det_inv = @intCast(i);
                break;
            }
        }
        if (det_inv == null) return HillCipherError.InvalidDeterminant;

        const inv = [_][2]i64{
            .{ @mod(det_inv.? * d, MODULUS), @mod(det_inv.? * (-b), MODULUS) },
            .{ @mod(det_inv.? * (-c), MODULUS), @mod(det_inv.? * a, MODULUS) },
        };

        return inv;
    }

    pub fn decrypt(self: HillCipher, allocator: Allocator, text: []const u8) ![]u8 {
        const decrypt_key = try self.makeDecryptKey();
        const clean = try self.processText(allocator, text);
        defer allocator.free(clean);

        const out = try allocator.alloc(u8, clean.len);
        errdefer allocator.free(out);

        var i: usize = 0;
        while (i < clean.len) : (i += 2) {
            const v0: i64 = @intCast(try self.replaceLetters(clean[i]));
            const v1: i64 = @intCast(try self.replaceLetters(clean[i + 1]));

            const d0 = @mod(decrypt_key[0][0] * v0 + decrypt_key[0][1] * v1, MODULUS);
            const d1 = @mod(decrypt_key[1][0] * v0 + decrypt_key[1][1] * v1, MODULUS);

            out[i] = self.replaceDigits(d0);
            out[i + 1] = self.replaceDigits(d1);
        }

        return out;
    }
};

fn gcd(a_init: i64, b_init: i64) i64 {
    var a = if (a_init < 0) -a_init else a_init;
    var b = if (b_init < 0) -b_init else b_init;
    while (b != 0) {
        const t = @mod(a, b);
        a = b;
        b = t;
    }
    return a;
}

test "hill cipher: python samples" {
    const alloc = testing.allocator;

    const hc = try HillCipher.init(.{ .{ 2, 5 }, .{ 1, 6 } });

    try testing.expectEqual(@as(usize, 19), try hc.replaceLetters('T'));
    try testing.expectEqual(@as(usize, 26), try hc.replaceLetters('0'));
    try testing.expectEqual(@as(u8, 'T'), hc.replaceDigits(19));
    try testing.expectEqual(@as(u8, '0'), hc.replaceDigits(26));

    const processed = try hc.processText(alloc, "Testing Hill Cipher");
    defer alloc.free(processed);
    try testing.expectEqualStrings("TESTINGHILLCIPHERR", processed);

    const enc = try hc.encrypt(alloc, "testing hill cipher");
    defer alloc.free(enc);
    try testing.expectEqualStrings("WHXYJOLM9C6XT085LL", enc);

    const dec = try hc.decrypt(alloc, "WHXYJOLM9C6XT085LL");
    defer alloc.free(dec);
    try testing.expectEqualStrings("TESTINGHILLCIPHERR", dec);
}

test "hill cipher: hello sample and invalid determinant" {
    const alloc = testing.allocator;

    const hc = try HillCipher.init(.{ .{ 2, 5 }, .{ 1, 6 } });
    const enc = try hc.encrypt(alloc, "hello");
    defer alloc.free(enc);
    try testing.expectEqualStrings("85FF00", enc);

    const dec = try hc.decrypt(alloc, "85FF00");
    defer alloc.free(dec);
    try testing.expectEqualStrings("HELLOO", dec);

    try testing.expectError(HillCipherError.InvalidDeterminant, HillCipher.init(.{ .{ 2, 4 }, .{ 1, 2 } }));
}

test "hill cipher: extreme long round trip" {
    const alloc = testing.allocator;
    const hc = try HillCipher.init(.{ .{ 2, 5 }, .{ 1, 6 } });

    const n: usize = 10000;
    const msg = try alloc.alloc(u8, n);
    defer alloc.free(msg);

    for (msg, 0..) |*ch, i| {
        ch.* = switch (i % 5) {
            0 => 'A',
            1 => '9',
            2 => ' ',
            3 => '#',
            else => 'b',
        };
    }

    const enc = try hc.encrypt(alloc, msg);
    defer alloc.free(enc);
    const dec = try hc.decrypt(alloc, enc);
    defer alloc.free(dec);

    const norm = try hc.processText(alloc, msg);
    defer alloc.free(norm);
    try testing.expectEqualSlices(u8, norm, dec);
}
