//! Base32 Encoding/Decoding - Zig implementation
//! Reference: https://github.com/TheAlgorithms/Python/blob/master/ciphers/base32.py

const std = @import("std");
const testing = std.testing;
const Allocator = std.mem.Allocator;

pub const Base32Error = error{
    InvalidUtf8,
    InvalidCharacter,
};

const B32_CHARSET = "ABCDEFGHIJKLMNOPQRSTUVWXYZ234567";

fn appendBinaryZfill8(bits: *std.ArrayListUnmanaged(u8), allocator: Allocator, codepoint: u21) !void {
    var buf: [32]u8 = undefined;
    var len: usize = 0;
    var value: u32 = codepoint;

    if (value == 0) {
        buf[0] = '0';
        len = 1;
    } else {
        while (value > 0) {
            buf[len] = if ((value & 1) == 1) '1' else '0';
            value >>= 1;
            len += 1;
        }

        var i: usize = 0;
        while (i < len / 2) : (i += 1) {
            std.mem.swap(u8, &buf[i], &buf[len - 1 - i]);
        }
    }

    if (len < 8) {
        try bits.appendNTimes(allocator, '0', 8 - len);
    }
    try bits.appendSlice(allocator, buf[0..len]);
}

fn bitsToInt(bits: []const u8) u8 {
    var value: u8 = 0;
    for (bits) |b| {
        value = (value << 1) | (if (b == '1') @as(u8, 1) else @as(u8, 0));
    }
    return value;
}

/// Encodes bytes using Python-reference base32 behavior.
/// Note: for empty input, Python reference returns `A=======`.
/// Time complexity: O(n), Space complexity: O(n)
pub fn base32Encode(allocator: Allocator, data: []const u8) ![]u8 {
    if (!std.unicode.utf8ValidateSlice(data)) return Base32Error.InvalidUtf8;

    var bits = std.ArrayListUnmanaged(u8){};
    defer bits.deinit(allocator);

    var it = std.unicode.Utf8View.initUnchecked(data).iterator();
    while (it.nextCodepoint()) |cp| {
        try appendBinaryZfill8(&bits, allocator, cp);
    }

    const padded_len = 5 * ((bits.items.len / 5) + 1);
    if (padded_len > bits.items.len) {
        try bits.appendNTimes(allocator, '0', padded_len - bits.items.len);
    }

    var b32 = std.ArrayListUnmanaged(u8){};
    defer b32.deinit(allocator);

    var i: usize = 0;
    while (i < bits.items.len) : (i += 5) {
        const idx = bitsToInt(bits.items[i .. i + 5]);
        try b32.append(allocator, B32_CHARSET[idx]);
    }

    const result_len = 8 * ((b32.items.len / 8) + 1);
    const out = try allocator.alloc(u8, result_len);
    errdefer allocator.free(out);

    @memcpy(out[0..b32.items.len], b32.items);
    @memset(out[b32.items.len..], '=');
    return out;
}

/// Decodes bytes using Python-reference base32 behavior.
/// Time complexity: O(n), Space complexity: O(n)
pub fn base32Decode(allocator: Allocator, data: []const u8) ![]u8 {
    var stripped_end = data.len;
    while (stripped_end > 0 and data[stripped_end - 1] == '=') stripped_end -= 1;
    const stripped = data[0..stripped_end];

    var bits = std.ArrayListUnmanaged(u8){};
    defer bits.deinit(allocator);

    for (stripped) |ch| {
        const idx_opt = std.mem.indexOfScalar(u8, B32_CHARSET, ch);
        if (idx_opt == null) return Base32Error.InvalidCharacter;
        const idx: u8 = @intCast(idx_opt.?);

        var bit: i32 = 4;
        while (bit >= 0) : (bit -= 1) {
            const one = ((idx >> @intCast(bit)) & 1) == 1;
            try bits.append(allocator, if (one) '1' else '0');
        }
    }

    const byte_count = bits.items.len / 8;
    const out = try allocator.alloc(u8, byte_count);
    errdefer allocator.free(out);

    for (0..byte_count) |k| {
        out[k] = bitsToInt(bits.items[k * 8 .. k * 8 + 8]);
    }
    return out;
}

test "base32: python samples" {
    const alloc = testing.allocator;

    const a = try base32Encode(alloc, "Hello World!");
    defer alloc.free(a);
    try testing.expectEqualStrings("JBSWY3DPEBLW64TMMQQQ====", a);

    const b = try base32Encode(alloc, "123456");
    defer alloc.free(b);
    try testing.expectEqualStrings("GEZDGNBVGY======", b);

    const c = try base32Encode(alloc, "some long complex string");
    defer alloc.free(c);
    try testing.expectEqualStrings("ONXW2ZJANRXW4ZZAMNXW24DMMV4CA43UOJUW4ZY=", c);

    const da = try base32Decode(alloc, "JBSWY3DPEBLW64TMMQQQ====");
    defer alloc.free(da);
    try testing.expectEqualStrings("Hello World!", da);
}

test "base32: python empty behavior" {
    const alloc = testing.allocator;

    const e = try base32Encode(alloc, "");
    defer alloc.free(e);
    try testing.expectEqualStrings("A=======", e);

    const d = try base32Decode(alloc, "");
    defer alloc.free(d);
    try testing.expectEqual(@as(usize, 0), d.len);
}

test "base32: invalid character" {
    const alloc = testing.allocator;
    try testing.expectError(Base32Error.InvalidCharacter, base32Decode(alloc, "ABC$"));
}

test "base32: extreme round-trip ascii" {
    const alloc = testing.allocator;
    const n: usize = 7000;
    const bytes = try alloc.alloc(u8, n);
    defer alloc.free(bytes);
    for (0..n) |i| bytes[i] = @intCast(32 + (i % 90));

    const enc = try base32Encode(alloc, bytes);
    defer alloc.free(enc);
    const dec = try base32Decode(alloc, enc);
    defer alloc.free(dec);

    try testing.expectEqualSlices(u8, bytes, dec);
}
