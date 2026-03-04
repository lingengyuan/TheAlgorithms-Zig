//! ROT13 / Caesar Shift Cipher - Zig implementation
//! Reference: https://github.com/TheAlgorithms/Python/blob/master/ciphers/rot13.py

const std = @import("std");
const testing = std.testing;
const Allocator = std.mem.Allocator;

/// Applies Caesar shift to ASCII letters; non-letters are unchanged.
/// Using shift=13 gives ROT13 behavior.
/// Time complexity: O(n), Space complexity: O(n)
pub fn dencrypt(allocator: Allocator, input: []const u8, shift: i64) ![]u8 {
    const out = try allocator.alloc(u8, input.len);
    errdefer allocator.free(out);

    const normalized = @mod(shift, @as(i64, 26));

    for (input, 0..) |ch, i| {
        if (ch >= 'A' and ch <= 'Z') {
            const base: i64 = 'A';
            const value: i64 = @intCast(ch);
            const mapped = @mod(value - base + normalized, @as(i64, 26)) + base;
            out[i] = @intCast(mapped);
        } else if (ch >= 'a' and ch <= 'z') {
            const base: i64 = 'a';
            const value: i64 = @intCast(ch);
            const mapped = @mod(value - base + normalized, @as(i64, 26)) + base;
            out[i] = @intCast(mapped);
        } else {
            out[i] = ch;
        }
    }

    return out;
}

test "rot13: python sample" {
    const alloc = testing.allocator;
    const msg = "My secret bank account number is 173-52946 so don't tell anyone!!";
    const encrypted = try dencrypt(alloc, msg, 13);
    defer alloc.free(encrypted);
    try testing.expectEqualStrings(
        "Zl frperg onax nppbhag ahzore vf 173-52946 fb qba'g gryy nalbar!!",
        encrypted,
    );

    const decrypted = try dencrypt(alloc, encrypted, 13);
    defer alloc.free(decrypted);
    try testing.expectEqualStrings(msg, decrypted);
}

test "rot13: custom shift and non letters" {
    const alloc = testing.allocator;
    const out = try dencrypt(alloc, "Abc-XYZ 123", 5);
    defer alloc.free(out);
    try testing.expectEqualStrings("Fgh-CDE 123", out);
}

test "rot13: empty input" {
    const alloc = testing.allocator;
    const out = try dencrypt(alloc, "", 13);
    defer alloc.free(out);
    try testing.expectEqual(@as(usize, 0), out.len);
}

test "rot13: extreme shift round-trip" {
    const alloc = testing.allocator;
    const input = "Please don't brute force me!";

    const a = try dencrypt(alloc, input, std.math.maxInt(i64));
    defer alloc.free(a);
    const b = try dencrypt(alloc, a, -std.math.maxInt(i64));
    defer alloc.free(b);
    try testing.expectEqualStrings(input, b);
}
