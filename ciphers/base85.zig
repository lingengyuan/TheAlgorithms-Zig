//! Base85 (Ascii85-like) Encoding/Decoding - Zig implementation
//! Reference: https://github.com/TheAlgorithms/Python/blob/master/ciphers/base85.py

const std = @import("std");
const testing = std.testing;
const Allocator = std.mem.Allocator;

pub const Base85Error = error{
    InvalidUtf8,
    InvalidCharacter,
};

fn base10To85(allocator: Allocator, d_init: u64) ![]u8 {
    var d = d_init;
    var list = std.ArrayListUnmanaged(u8){};
    errdefer list.deinit(allocator);

    while (d > 0) {
        try list.append(allocator, @intCast((d % 85) + 33));
        d /= 85;
    }
    return try list.toOwnedSlice(allocator);
}

fn base85To10(digits: []const u8) u64 {
    var sum: u64 = 0;
    for (digits, 0..) |digit, i| {
        var pow: u64 = 1;
        for (0..(digits.len - 1 - i)) |_| pow *= 85;
        sum += @as(u64, digit) * pow;
    }
    return sum;
}

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

    if (len < 8) try bits.appendNTimes(allocator, '0', 8 - len);
    try bits.appendSlice(allocator, buf[0..len]);
}

fn bitsToInt(bits: []const u8) u32 {
    var value: u32 = 0;
    for (bits) |b| value = (value << 1) | (if (b == '1') @as(u32, 1) else @as(u32, 0));
    return value;
}

/// Encodes bytes using Python-reference base85 implementation behavior.
/// Time complexity: O(n), Space complexity: O(n)
pub fn ascii85Encode(allocator: Allocator, data: []const u8) ![]u8 {
    if (!std.unicode.utf8ValidateSlice(data)) return Base85Error.InvalidUtf8;

    var bits = std.ArrayListUnmanaged(u8){};
    defer bits.deinit(allocator);

    var it = std.unicode.Utf8View.initUnchecked(data).iterator();
    while (it.nextCodepoint()) |cp| {
        try appendBinaryZfill8(&bits, allocator, cp);
    }

    const padded_bit_len = 32 * ((bits.items.len / 32) + 1);
    const null_values: usize = (padded_bit_len - bits.items.len) / 8;
    if (padded_bit_len > bits.items.len) {
        try bits.appendNTimes(allocator, '0', padded_bit_len - bits.items.len);
    }

    var result = std.ArrayListUnmanaged(u8){};
    defer result.deinit(allocator);

    var i: usize = 0;
    while (i < bits.items.len) : (i += 32) {
        const chunk = bitsToInt(bits.items[i .. i + 32]);
        const encoded = try base10To85(allocator, chunk);
        defer allocator.free(encoded);

        // Python reverses the generated digits.
        var j = encoded.len;
        while (j > 0) {
            j -= 1;
            try result.append(allocator, encoded[j]);
        }
    }

    const out_slice = if (null_values % 4 != 0 and result.items.len >= null_values)
        result.items[0 .. result.items.len - null_values]
    else
        result.items;

    const out = try allocator.alloc(u8, out_slice.len);
    @memcpy(out, out_slice);
    return out;
}

/// Decodes bytes using Python-reference base85 implementation behavior.
/// Time complexity: O(n), Space complexity: O(n)
pub fn ascii85Decode(allocator: Allocator, data: []const u8) ![]u8 {
    const null_values: usize = 5 * ((data.len / 5) + 1) - data.len;

    var padded = std.ArrayListUnmanaged(u8){};
    defer padded.deinit(allocator);
    try padded.appendSlice(allocator, data);
    try padded.appendNTimes(allocator, 'u', null_values);

    if (padded.items.len % 5 != 0) return Base85Error.InvalidCharacter;

    var result_chars = std.ArrayListUnmanaged(u21){};
    defer result_chars.deinit(allocator);

    var i: usize = 0;
    while (i < padded.items.len) : (i += 5) {
        var digits: [5]u8 = undefined;
        for (0..5) |k| {
            const c = padded.items[i + k];
            if (c < 33 or c > 117) return Base85Error.InvalidCharacter;
            digits[k] = c - 33;
        }

        const value = base85To10(&digits);

        var bits = value;
        var bytes4: [4]u8 = undefined;
        var idx: i32 = 3;
        while (idx >= 0) : (idx -= 1) {
            bytes4[@intCast(idx)] = @intCast(bits & 0xff);
            bits >>= 8;
        }

        for (bytes4) |b| try result_chars.append(allocator, b);
    }

    const offset: usize = if (null_values % 5 == 0) 1 else 0;
    const end = if (offset >= null_values) offset - null_values else result_chars.items.len - (null_values - offset);
    const trimmed = result_chars.items[0..end];

    // Python does bytes(result, "utf-8").
    var out = std.ArrayListUnmanaged(u8){};
    errdefer out.deinit(allocator);

    for (trimmed) |cp| {
        var buf: [4]u8 = undefined;
        const n = try std.unicode.utf8Encode(cp, &buf);
        try out.appendSlice(allocator, buf[0..n]);
    }

    return try out.toOwnedSlice(allocator);
}

test "base85: python samples" {
    const alloc = testing.allocator;

    const e0 = try ascii85Encode(alloc, "");
    defer alloc.free(e0);
    try testing.expectEqualStrings("", e0);

    const e1 = try ascii85Encode(alloc, "12345");
    defer alloc.free(e1);
    try testing.expectEqualStrings("0etOA2#", e1);

    const e2 = try ascii85Encode(alloc, "base 85");
    defer alloc.free(e2);
    try testing.expectEqualStrings("@UX=h+?24", e2);

    const d1 = try ascii85Decode(alloc, "0etOA2#");
    defer alloc.free(d1);
    try testing.expectEqualStrings("12345", d1);
}

test "base85: decode empty" {
    const alloc = testing.allocator;
    const d = try ascii85Decode(alloc, "");
    defer alloc.free(d);
    try testing.expectEqual(@as(usize, 0), d.len);
}

test "base85: invalid input character" {
    const alloc = testing.allocator;
    try testing.expectError(Base85Error.InvalidCharacter, ascii85Decode(alloc, "~{"));
}

test "base85: extreme round-trip ascii" {
    const alloc = testing.allocator;
    const n: usize = 5000;
    const data = try alloc.alloc(u8, n);
    defer alloc.free(data);
    for (0..n) |i| data[i] = @intCast(32 + (i % 90));

    const enc = try ascii85Encode(alloc, data);
    defer alloc.free(enc);
    const dec = try ascii85Decode(alloc, enc);
    defer alloc.free(dec);

    try testing.expectEqualSlices(u8, data, dec);
}
